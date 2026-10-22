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

<!-- RESULTS -->

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
