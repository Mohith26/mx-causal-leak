"""Measure future leakage through MXFP4-quantized V, and check that the block-diagonal fix removes it.

    python experiments.py
"""

import torch

from mxleak.core import causal_attention, mxfp4

BLOCK = 32


def causality_check(scale_mode, t=256, d=64, seed=0):
    """Max change in outputs at positions <= i when only positions > i change."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(t, d, generator=g, dtype=torch.float64) for _ in range(3))
    res = {}
    for mode in ("float", "leaky", "fixed"):
        worst = 0.0
        for i in range(5, t, 37):  # positions at various offsets inside their block
            v2 = v.clone()
            v2[i + 1:] += torch.randn(t - i - 1, d, generator=g, dtype=torch.float64) * 8
            a = causal_attention(q, k, v, mode, BLOCK, scale_mode)[: i + 1]
            b = causal_attention(q, k, v2, mode, BLOCK, scale_mode)[: i + 1]
            worst = max(worst, (a - b).abs().max().item())
        res[mode] = worst
    return res


def outlier_effect(scale_mode, mode, n=300, t=64, d=32, outlier=6.0, seed=1):
    """Add a 6x outlier to the value at position 31 (last token of block 0) and look at the
    outputs at positions 0-30, which must not depend on it. Returns the share of those
    outputs that change at all and the median relative change among the ones that do."""
    g = torch.Generator().manual_seed(seed)
    changed, rel = [], []
    for _ in range(n):
        q, k, v = (torch.randn(t, d, generator=g, dtype=torch.float64) for _ in range(3))
        v2 = v.clone()
        v2[BLOCK - 1] *= outlier
        a = causal_attention(q, k, v, mode, BLOCK, scale_mode)[: BLOCK - 1]
        b = causal_attention(q, k, v2, mode, BLOCK, scale_mode)[: BLOCK - 1]
        diff = (a - b).norm(dim=1)
        changed.append(diff > 1e-12)
        rel.append(diff / a.norm(dim=1))
    changed, rel = torch.cat(changed), torch.cat(rel)
    med = rel[changed].median().item() if changed.any() else 0.0
    return changed.double().mean().item(), med


def underflow_rate(scale_mode, outlier=6.0, n=2000, seed=2):
    """Share of the other 31 values in a V block that quantize to exactly 0, without and
    with one outlier at the end of the block."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, BLOCK, generator=g, dtype=torch.float64)
    base = (mxfp4(v, 1, BLOCK, scale_mode)[:, :-1] == 0).double().mean().item()
    v[:, -1] = v[:, -1].sign() * v.abs().amax(1) * outlier
    out = (mxfp4(v, 1, BLOCK, scale_mode)[:, :-1] == 0).double().mean().item()
    return base, out


def float_fraction(t, block=BLOCK):
    """Fraction of causal (unmasked) P x V multiply-adds that the fix runs unquantized."""
    causal = t * (t + 1) / 2
    diag = (t // block) * block * (block + 1) / 2
    return diag / causal


def main():
    for scale_mode in ("maxabs", "ocp"):
        print(f"\n## scale_mode = {scale_mode}\n")
        c = causality_check(scale_mode)
        print("Max |change| in outputs at positions <= i after changing only positions > i "
              "(T=256, d=64, 7 cut points):")
        for m, val in c.items():
            print(f"  {m:6s} {val:.3e}")
        base, out = underflow_rate(scale_mode)
        print(f"Earlier values in a V block that round to 0: {base:.1%} normally, "
              f"{out:.1%} when the block's last token is a 6x outlier")
        print("6x value outlier at position 31; outputs at positions 0-30 (300 sequences, T=64, d=32):")
        for m in ("float", "leaky", "fixed"):
            share, med = outlier_effect(scale_mode, m)
            print(f"  {m:6s} {share:6.1%} of outputs change, median relative change {med:.1%}")
    print("\nShare of causal P x V work the fix keeps in float:")
    for t in (1024, 4096, 32768):
        print(f"  T={t:6d}: {float_fraction(t):.2%}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
