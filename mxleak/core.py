"""MXFP4 block quantization and causal attention with block-quantized P x V.

Reproduces the setup from MatX's "Future leakage in block-quantized attention"
(Jan 2026): V is quantized in blocks that run along the token axis, so a value
at position j shares a scale with later positions in its block.
"""

from __future__ import annotations

import math

import torch

# E2M1 (FP4) magnitudes
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4(x: torch.Tensor, dim: int, block: int = 32, scale_mode: str = "maxabs") -> torch.Tensor:
    """Fake-quantize x to FP4 with one shared scale per `block` consecutive elements along `dim`.

    scale_mode="maxabs": scale = amax / 6 (the block max maps exactly to the top of the grid).
    scale_mode="ocp":    scale = 2 ** (floor(log2(amax)) - 2), the power-of-two scale of the
                         OCP MX spec (values above 6 * scale saturate).
    Returns the dequantized tensor (same shape and dtype as x).
    """
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    assert n % block == 0, f"length {n} is not a multiple of block {block}"
    xb = x.reshape(*x.shape[:-1], n // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    tiny = torch.finfo(x.dtype).tiny
    if scale_mode == "maxabs":
        scale = amax / 6.0
    elif scale_mode == "ocp":
        scale = torch.exp2(torch.floor(torch.log2(amax.clamp_min(tiny))) - 2)
    else:
        raise ValueError(scale_mode)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    y = (xb / scale).clamp(-6.0, 6.0)
    grid = FP4_GRID.to(x.dtype)
    idx = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)  # round to nearest grid point
    q = torch.sign(y) * grid[idx] * scale
    return q.reshape(*x.shape).movedim(-1, dim)


def causal_attention(q, k, v, mode: str = "float", block: int = 32, scale_mode: str = "maxabs"):
    """Single-head causal attention. q, k, v: (T, d).

    mode="float": no quantization.
    mode="leaky": Q, K quantized along d; P and V quantized along the token axis; every
                  P-tile x V-block product uses the quantized operands.
    mode="fixed": same, except the block-diagonal tiles (query block == key block) use
                  unquantized P and V, which is the fix proposed in the post.
    """
    t, d = q.shape
    if mode == "float":
        s = q @ k.T / math.sqrt(d)
    else:
        s = mxfp4(q, 1, block, scale_mode) @ mxfp4(k, 1, block, scale_mode).T / math.sqrt(d)
    mask = torch.ones(t, t, dtype=torch.bool).tril()
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
    if mode == "float":
        return p @ v
    pq = mxfp4(p, 1, block, scale_mode)  # blocks along key positions
    vq = mxfp4(v, 0, block, scale_mode)  # blocks along token positions
    if mode == "leaky":
        return pq @ vq
    if mode != "fixed":
        raise ValueError(mode)
    nb = t // block
    out = torch.zeros(t, v.shape[1], dtype=v.dtype)
    for qb in range(nb):
        rows = slice(qb * block, (qb + 1) * block)
        for kb in range(qb + 1):
            cols = slice(kb * block, (kb + 1) * block)
            if kb == qb:
                out[rows] += p[rows, cols] @ v[cols]
            else:
                out[rows] += pq[rows, cols] @ vq[cols]
    return out
