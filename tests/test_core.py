import math

import pytest
import torch

from mxleak.core import FP4_GRID, causal_attention, mxfp4

D = torch.float64


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=D)


def test_representable_values_round_trip():
    # block max 6 -> scale 1, so every grid value is exact
    x = torch.cat([FP4_GRID, -FP4_GRID]).to(D).repeat(2)[:32]
    x[0] = 6.0
    assert torch.equal(mxfp4(x[None], 1), x[None])


def test_maxabs_maps_block_max_to_top_of_grid():
    x = rand(4, 64)
    q = mxfp4(x, 1, scale_mode="maxabs")
    for b in range(2):
        blk, qb = x[:, b * 32:(b + 1) * 32], q[:, b * 32:(b + 1) * 32]
        i = blk.abs().argmax(1)
        assert torch.allclose(qb.gather(1, i[:, None]), blk.gather(1, i[:, None]))


def test_ocp_scale_is_power_of_two():
    x = rand(1, 32) * 3.7
    q = mxfp4(x, 1, scale_mode="ocp")
    amax = x.abs().max()
    scale = 2.0 ** (math.floor(math.log2(amax)) - 2)
    ratios = (q.abs() / scale).flatten().tolist()
    assert set(ratios) <= set(FP4_GRID.tolist())


def test_blocks_are_independent_along_dim():
    x = rand(64, 3)
    y = x.clone()
    y[40] += 100.0  # only the second block along dim 0 changes
    assert torch.equal(mxfp4(x, 0)[:32], mxfp4(y, 0)[:32])
    assert not torch.equal(mxfp4(x, 0)[32:], mxfp4(y, 0)[32:])


@pytest.mark.parametrize("scale_mode", ["maxabs", "ocp"])
def test_leaky_breaks_causality_and_fixed_does_not(scale_mode):
    t, d = 128, 32
    q, k, v = rand(t, d, seed=1), rand(t, d, seed=2), rand(t, d, seed=3)
    i = 70  # in the middle of block 2
    v2 = v.clone()
    v2[i + 1:] *= 10
    for mode, should_leak in (("float", False), ("leaky", True), ("fixed", False)):
        a = causal_attention(q, k, v, mode, 32, scale_mode)[: i + 1]
        b = causal_attention(q, k, v2, mode, 32, scale_mode)[: i + 1]
        assert (not torch.equal(a, b)) == should_leak, mode


def test_fixed_matches_leaky_outside_the_diagonal_tiles():
    t, d = 96, 32
    q, k, v = rand(t, d, seed=4), rand(t, d, seed=5), rand(t, d, seed=6)
    s = mxfp4(q, 1) @ mxfp4(k, 1).T / math.sqrt(d)
    p = torch.softmax(s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf")), -1)
    pq, vq = mxfp4(p, 1), mxfp4(v, 0)
    diag_q = torch.zeros(t, d, dtype=D)
    diag_f = torch.zeros(t, d, dtype=D)
    for b in range(t // 32):
        r = slice(b * 32, (b + 1) * 32)
        diag_q[r] = pq[r, r] @ vq[r]
        diag_f[r] = p[r, r] @ v[r]
    leaky = causal_attention(q, k, v, "leaky")
    fixed = causal_attention(q, k, v, "fixed")
    assert torch.allclose(fixed - diag_f, leaky - diag_q, atol=1e-12)
