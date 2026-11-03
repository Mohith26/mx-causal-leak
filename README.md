# mx-causal-leak

A small, runnable reproduction of the mechanism in MatX's post [Future leakage in block-quantized attention](https://matx.com/research/leaky_quantization) (Jan 2026), plus a check that their fix closes it.

The idea from the post: to use block-quantized (MXFP4) matmuls for `P x V`, V has to be quantized in blocks of 32 that run along the token axis. Each block shares one scale, and the scale depends on every token in the block, including later ones. So inside the block-diagonal tile, the output at position i can depend on tokens after i. Their fix is to keep the block-diagonal tiles of `P x V` unquantized and quantize everything else.

I implemented fake-quantized MXFP4 (E2M1 grid, one scale per 32 elements, either `amax / 6` or the OCP power-of-two scale), single-head causal attention in three modes (`float`, `leaky`, `fixed`), and measured the leak directly instead of through a training run.

## Results

`python experiments.py` (float64, CPU, a few seconds):

| | float | leaky | fixed |
|---|---|---|---|
| max change in outputs at positions <= i when only positions > i change (T=256, d=64) | 0 | 1.37 | 0 |
| outputs at positions 0-30 that change when position 31 gets a 6x value outlier | 0% | 100% | 0% |
| median relative change of those outputs | 0% | 16.9% | 0% |

(`amax / 6` scales; the OCP power-of-two scale gives 1.06, 100% and 16.1%.)

- **The leak is real and not subtle at the output level.** With one 6x outlier at the end of a block, every earlier output in that block moves, by about 17% of its norm (median).
- **The channel is underflow, as the post says.** Normally 7.7% of a V block's values round to zero in FP4; with a 6x outlier at the end of the block, 43.7% of the other 31 do.
- **The fix is exact.** With the block-diagonal tiles in float, outputs at positions <= i are bit-for-bit unchanged no matter what happens after i, and everything outside those tiles is identical to the fully quantized version (tested).
- **The fix is cheap at real context lengths.** The share of causal `P x V` multiply-adds it leaves unquantized is 3.2% at T=1024, 0.8% at T=4096 and 0.1% at T=32768 (it's about block size / T).

What this doesn't do: it doesn't train a model, so it doesn't reproduce the post's parallel vs autoregressive loss gap. It only shows the causal dependence exists, how big it is on random inputs, and that the fix removes it exactly. It also uses fake quantization in PyTorch, not real FP4 kernels.

## Layout

- `mxleak/core.py`: `mxfp4()` fake quantization and `causal_attention(q, k, v, mode)`.
- `experiments.py`: the numbers above.
- `tests/`: 7 tests (rounding, scales, block independence, causality per mode, and that `fixed` differs from `leaky` only on the diagonal tiles).

```sh
pip install torch pytest
python -m pytest -q
python experiments.py
```
