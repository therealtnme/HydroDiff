# HydroDiff research notes

This file records what was tried, what was measured, and what was kept or dropped.

## Bench

* Data: CIFAR-10, 32x32, class-conditional (50,000 training images; the first 1,000 test images for validation).
* Model: `hydrodiff-s` (width 192, 6 blocks, 3 heads, 4x4 patches = 64 tokens, 3.07M parameters) unless stated.
* Training: 4,000 steps at batch 64 (about 5 epochs), Muon + AdamW, warmup-stable-decay schedule, EMA 0.999, one CPU
  thread per run, four runs in parallel. This is a very short budget: the numbers compare ideas against each other and
  are far from what the architecture reaches with a full training run.
* Validation (`test/validate.py`): fixed images, noise and noise levels for every checkpoint.
  * `v_uniform`: the training loss with one noise level per image.
  * `v_mixed`: the same with per-token noise levels (token speeds in [1, 2]).
  * `x0_mse@a-b`: unweighted error of the clean-image estimate for noise levels in [a, b).
* Sampling quality (`test/evaluate.py`): FID of generated samples against 10,000 training images (InceptionV3 pool
  features). With fewer than 50,000 samples FID is biased upwards, so numbers are only compared at the same sample count.
* Speed: `compute` is the number of token evaluations divided by `steps * tokens` (what a dense sampler spends);
  `ms / image` is the measured CPU wall-clock time.

FID noise: the same setting with a different sampling seed moved by 0.6-1.2 FID at 2,000 samples (51.99 / 51.36,
51.96 / 51.05, 49.27 / 48.09, 53.76 / 52.71). Every setting uses the same starting noise for the same seed, so
differences between settings are paired and somewhat less noisy than that, but differences under about 1 FID are not
evidence of anything.

<!-- SUMMARY -->

## 1. Mixed noise levels in training (`hetero_prob`)

The adaptive sampler puts tokens at different noise levels, so the model must be trained on such states. `base` trains
half of its images that way (`hetero_prob=0.5`, token speeds in [1, 2]); `nohetero` trains only on one level per image.

| Model | v_uniform | v_mixed | x0_mse t<0.1 | t 0.1-0.3 | t 0.3-0.6 | t 0.6-1 |
| --- | --- | --- | --- | --- | --- | --- |
| base (`hetero_prob=0.5`) | **0.1385** | **0.1314** | **0.00077** | **0.00577** | 0.02117 | 0.11307 |
| nohetero (`hetero_prob=0`) | 0.1444 | 0.1577 | 0.00096 | 0.00583 | 0.02115 | **0.11054** |

Mixed-level training makes the model better on mixed states (as it must) and 20% better at low noise, where the
cleaner neighbours help. It is 2.3% worse at high noise, and high noise decides the global structure that FID sees:

| FID (2,000 samples), Euler | 8 steps | 12 steps | 16 steps | 32 steps |
| --- | --- | --- | --- | --- |
| nohetero, dense sampler | **52.09** | **51.17** | **50.79** | **49.27** / 48.09 |
| base, dense sampler | 54.87 | 52.76 | 51.99 / 51.36 | 51.96 / 51.05 |

## 2. Noise mask (per-token speed from the uncertainty head)

### What the mask sees

`test/mask_analysis.py` ranks each token's predicted error against the detail of its clean patch (500 test images,
Spearman correlation per image, averaged):

| Noise level | vs high-pass energy | vs pixel standard deviation |
| --- | --- | --- |
| t = 0.3 | 0.67 | 0.39 |
| t = 0.6 | 0.57 | 0.33 |
| t = 0.9 | 0.25 | 0.12 |

The head finds the fine, unpredictable detail - the user's "fire, grain, dust, leaves, cracks" - not just edges or
contrast, and it does so long before the detail is visible (t = 0.6). In generated images, flat backgrounds and sky
finish first and objects and texture finish last:

![Generated images (top) and the step at which each token finished (bottom: dark = first, bright = last)](figures/noise_mask.png)

### Accuracy and compute

On the model trained for it (`base`), the noise mask is better than the dense sampler at every step count from 12 up,
with less compute, and keeps improving where the dense sampler has stopped:

| Euler, base model | FID dense | FID noise mask | compute of the noise mask |
| --- | --- | --- | --- |
| 8 steps | **54.87** | 55.85 | 0.96 |
| 10 steps | - | 53.44 | 0.94 |
| 12 steps | 52.76 | **52.44** | 0.91 |
| 16 steps | 51.99 / 51.36 | **51.03** | 0.89 |
| 20 steps | - | **50.33** | 0.87 |
| 32 steps | 51.96 / 51.05 | **49.67 / 48.91** | 0.84 |

At 32 steps the gain is 2.3 and 2.1 FID on two sampling seeds. Below about 10 steps the warmup leaves too few steps for
tokens to separate, and the mask neither helps nor saves compute.

On a model **not** trained on mixed states the mask hurts: `nohetero` goes from 49.27 to 50.20 (Euler, 32 steps) and
from 50.13 to 52.92 (DPM-Solver, 32 steps).

**Net result.** Mixed-level training costs about as much FID with the dense sampler (2-3) as the noise mask then wins
back, so at this budget `base` + noise mask (49.67 / 48.91 at 0.84 compute) is level with `nohetero` + dense sampler
(49.27 / 48.09 at full compute): the same quality for 16% less compute, not better quality. Sections 6-8 test whether
that training cost can be removed.

### Sampler settings

| Setting (DPM-Solver, 32 steps, base model) | FID | compute |
| --- | --- | --- |
| noise mask, `warmup=0.25` (default) | 53.08 | 0.84 |
| `warmup=0.1` | 53.08 | 0.81 |
| `warmup=0.5` | 53.67 | 0.91 |
| `max_rate=3` (trained with 2) | 50.84 | 0.74 |
| Euler, `max_rate=3` | 49.58 | 0.74 |
| Euler, `max_rate=3`, 16 steps | 52.24 (2: 51.03) | 0.79 |
| Euler, `max_rate=3`, 12 steps | 54.43 (2: 52.44) | 0.82 |

A wider speed spread at sampling (`max_rate=3`) keeps the quality at 32 steps for 12% less compute, but hurts at 12-16
steps, where there are too few steps to absorb a token that runs three times as fast.

## 3. Update mask

### Skipping finished tokens: the key/value cache is free

| DPM-Solver, 32 steps, base model | FID | compute |
| --- | --- | --- |
| noise mask, finished tokens skipped (cache) | 53.08 | 0.84 |
| noise mask, every token computed every step (`skip_finished=False`) | 52.94 | 1.00 |
| noise mask, no `t = 0` refresh of finished tokens | 53.19 | 0.81 |

Not computing finished tokens - they attend only through cached keys and values - costs nothing measurable. The `t = 0`
refresh (one more evaluation per finished token) does not measurably help either.

### Reusing the surest tokens' estimates (`reuse`)

`reuse` takes the surest moving tokens out of a step (they move with their previous clean estimate). It needs no
special training, because the tokens stay on their own clocks:

| Euler, 32 steps | FID | compute | ms / image |
| --- | --- | --- | --- |
| nohetero, dense | 49.27 / 48.09 | 1.00 | 252 |
| nohetero, `reuse=0.25` | 48.24 | 0.82 | 217 |
| nohetero, `reuse=0.5` | 48.81 | **0.64** | **151** |
| nohetero, `reuse=0.5`, DPM-Solver | 48.49 | 0.64 | 158 |
| nohetero, `reuse=0.25`, DPM-Solver | **47.63** | 0.82 | 221 |
| base, dense | 51.96 / 51.05 | 1.00 | 233 |
| base, noise mask | 49.67 / 48.91 | 0.84 | 202 |
| base, noise mask + `reuse=0.25` | 50.02 | 0.71 | 173 |
| base, noise mask + `reuse=0.5` | 50.76 | 0.60 | 146 |
| base, noise mask + `reuse=0.5` + `max_rate=3` | 51.88 | 0.55 | 129 |
| base, `reuse=0.5` only | 51.65 | 0.64 | 152 |

The update mask is the clearest speed win: it saves a third of the compute (1.6-1.7x less wall-clock time) at the same FID.
At few steps it costs quality instead: base, Euler 16 steps, `reuse=0.25` gives 53.04 (compute 0.76) against 51.03 for
the noise mask alone (0.89) - no better than simply taking fewer steps.

(`ms / image` was measured with four evaluations sharing the CPU, one thread each; it tracks `compute` closely.)

## 4. Fresh noise per token (`churn`) - dropped

| Euler, 32 steps, base model | FID |
| --- | --- |
| noise mask | 49.67 |
| noise mask, `churn=0.2` | 49.68 |
| noise mask, `churn=0.5` | 52.19 |
| dense, `churn=0.2` | 51.93 (no churn: 51.96) |

Replacing part of the noise of unpredictable tokens with fresh noise does not help and hurts when strong. The
"how much noise where" idea works as a per-token noise *level* (section 2), not as per-token noise *injection*.
`churn` stays available but defaults to 0.

## 5. Solver: Euler beats DPM-Solver++(2M) here

| base model, dense | 8 steps | 16 steps | 32 steps |
| --- | --- | --- | --- |
| Euler | 54.87 | 51.99 | 51.96 |
| DPM-Solver++(2M) | 55.53 | 53.44 | 53.76 / 52.71 |

With a small, briefly trained clean-patch predictor the second-order extrapolation does not pay off, so `euler` stays
the default; `dpm2m` stays available.

## 6. Mixture-of-experts denoiser

All four models have the same active parameters per token: the MoE layers (blocks 1-5) hold one shared expert of 256
and eight routed experts of 256 with one active, against one dense MLP of 512. Sampling speed was measured alone on the
machine (4 threads); FID with DPM-Solver, 32 steps, 2,000 samples.

| Model | Params | Train img/s | v_uniform | v_mixed | FID dense | FID noise mask | ms / image (batch 250) dense / mask | Latency, batch 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 3.07M | **31.8** | 0.1383 | 0.1313 | 53.12 | 51.79 | 93 / 79 | **377 ms** |
| MoE, token routing | 8.23M | 22.3 | **0.1380** | **0.1311** | 52.80 | 51.02 | 90 / 85 | 593 ms |
| MoE, noise-level routing (`time`) | 8.23M | 22.4 | 0.1386 | 0.1315 | 49.86 | 51.09 | 92 / 81 | 379 ms |
| MoE, image routing | 8.23M | 22.9 | 0.1381 | 0.1312 | 53.89 | 51.89 | 92 / **74** | 386 ms |

At equal active compute the experts do not make the model more accurate at this scale - the validation losses are
within 0.3% of each other and the FID differences are inconsistent between the two samplers (noise-level routing is
best with the dense sampler and level with the others under the noise mask). They do not make it faster either:
training is 30% slower (expert dispatch on a CPU), batch sampling costs the same, and per-token routing makes
single-image latency 57% worse, because every expert is a separate small matrix product. Noise-level and image routing
keep the latency of the dense model, since at batch 1 nearly every token goes to the same expert in a step.

This matches the literature: token-level MoE wins at equal active FLOPs from B scale up, with long training on
accelerators. `num_experts=0` (dense) stays the default; the three routing modes stay available, and `time` routing is
the one to try first at a larger scale (no router, no balancing, no latency cost).

<!-- REMOTE -->

## Prior work

The three ideas each have close relatives in recent work. The review (2022-2026) shaped the design:

| Idea | Closest prior work | What HydroDiff takes from it |
| --- | --- | --- |
| Non-binary noise mask | Patch Forcing (CVPR 2026), AsyncPatch (2026), SRM (ICML 2025), Diffusion Forcing, MuLAN | Per-token noise levels with per-token conditioning; a cheap difficulty head; confident patches advance faster. Training states must look like sampling states: independent uniform levels per token train badly. |
| Binary update mask | RAS (CVPR 2026), ToCa, DuCa, DyDiT, DiffCR, SparseDiT, TREAD, MaskDiT | Gather only the active tokens; inactive tokens attend through cached keys and values; static shapes per step. |
| MoE denoiser | DiT-MoE, EC-DiT, DiffMoE, Race-DiT, DSMoE, Switch-DiT, DTR, eDiff-I | Fine-grained experts with a shared expert; the strongest small-scale evidence is for experts tied to the noise level, which motivated `moe_routing="time"`. |

Three warnings from the literature changed the user's original ideas:

1. **A learned noise allocation collapses.** A mask that is optimised against the denoising loss learns "add no noise"
   unless the mean noise level is fixed or an ELBO with a KL term is used (MuLAN). HydroDiff therefore does not learn
   where to put noise in training. The mask is applied at sampling time as a per-token *speed*, and training only has to
   cover the states the sampler can produce.
2. **A frozen token with a single global time is out of distribution.** RAS-style skipping reuses stale predictions
   and the model never saw frozen tokens in training. With per-token noise levels a skipped token is simply a token at
   its own noise level, which the model was trained on.
3. **MoE at tiny scale is unproven.** Token-level MoE beats dense models at equal active compute at B scale and above.
   At S scale the clear gains are from timestep specialisation. Both are measured here.

## Design decisions

### Noise mask = per-token speed

The user's description - "predict a mask that will determine how much noise is applied where" - can mean two things,
and both are implemented:

* **Per-token noise level** (`adaptive=True`, the default). The mask is each token's next noise level. A token's speed
  is `max_rate ** (1 - rank)`, where `rank` is the token's uncertainty rank among the tokens that still have noise.
  The surest token moves `max_rate` times faster than the schedule, the least sure one moves with the schedule and
  finishes on the last step.
* **Per-token fresh noise** (`churn`). A share of each token's noise is replaced with fresh noise in every step,
  scaled by the same rank, so the least predictable tokens are the most stochastic.

### Update mask = tokens that still change

* A token that reaches `t = 0` is never run again, so the update mask is exact: skipping it changes nothing about its
  own value. Its keys and values stay in the cache as context.
* `refresh_finished` runs a just-finished token once more at `t = 0`, so its cached keys and values describe the
  clean token, as the model saw clean tokens in training.
* `reuse` additionally lets the surest moving tokens skip one step with their last clean estimate (RAS-style), never
  two steps in a row and never on the step that finishes them.

### Bugs found by review agents

Two agents reviewed and tested the first version. Every training result in this file is from the corrected code.

| Defect | Effect | Fix |
| --- | --- | --- |
| adaLN-Zero init never ran | transformers skips custom-code modules without direct parameters, so the output projection and the modulation kept a random init | zero-init set on the linear layers themselves |
| Long skips connected each layer to itself | `long_skips=True` was only an extra linear map | only the first half of the blocks pushes |
| Top-1 router had no gradient | the normalised weight was exactly 1 | normalise only when `top_k > 1` |
| x-prediction target not clipped | a perfect prediction had loss 0.85 at `t = 0.01` | prediction and target share the clipped `1/t` |
| Image routing saw only active tokens at sampling | routing differed from training | cached router inputs for the whole image |
| FID crashed on SciPy 1.18 | `sqrtm(disp=...)` was removed | plain `sqrtm` with the finiteness fallback |
| `--compare` baseline was not dense | `reuse` stayed on | baseline keeps only `steps`, `solver`, `guidance_scale` |
