"""
Validation metrics with fixed definitions, so that every checkpoint is measured the same way whatever its config.

    python test/validate.py runs/base/ema runs/nohetero/ema

For 1000 test images with fixed noise:
- `v_uniform`: the training loss (velocity, x-prediction's clipped 1/t weighting) with one noise level per image, evenly
  spaced over (0, 1).
- `v_mixed`: the same with per-token levels: every image's tokens get speeds in [1, 2] (the adaptive sampler's states).
- `x0_mse@a-b`: mean squared error of the clean-image estimate for uniform levels in [a, b) - unweighted, so every noise
  band is visible on its own.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff import HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.modeling_hydrodiff import patchify  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_images, to_model_range  # noqa: E402

BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]


@torch.no_grad()
def metrics(model, images, labels, seed=1234, batch=250, max_rate=2.0):
    model.eval()
    cfg = model.config
    n = cfg.num_patches
    sums = {}

    def add(key, value, count):
        s, c = sums.get(key, (0.0, 0))
        sums[key] = (s + value * count, c + count)

    g = torch.Generator().manual_seed(seed)
    for i in range(0, len(images), batch):
        x = to_model_range(images[i : i + batch])
        lab = labels[i : i + batch]
        b = x.shape[0]
        x0 = patchify(x, cfg.patch_size)
        noise = torch.randn(b, n, cfg.patch_dim, generator=g)
        levels = ((torch.arange(b) + torch.rand(b, generator=g)) / b)[:, None]
        uniform = levels.expand(b, n)
        rate = 1 + (max_rate - 1) * torch.rand(b, n, generator=g)
        mixed = (1 - (1 - levels) * rate).clamp(0, 1)
        out = model(x, lab, timesteps=uniform, noise=noise)
        add("v_uniform", out.denoise_loss.item(), b)
        add("v_mixed", model(x, lab, timesteps=mixed, noise=noise).denoise_loss.item(), b)
        t = uniform[..., None]
        x_t = (1 - t) * x0 + t * noise
        x0_hat = out.prediction if cfg.prediction_type == "x" else x_t - t * out.prediction
        err = (x0_hat - x0).pow(2).mean((1, 2))
        for lo, hi in BANDS:
            sel = (levels[:, 0] >= lo) & (levels[:, 0] < hi)
            if sel.any():
                add(f"x0_mse@{lo}-{hi}", err[sel].mean().item(), int(sel.sum()))
    return {k: round(s / c, 5) for k, (s, c) in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--val-data", default="/home/user/data/cifar10_test.npz")
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    images, labels = load_images(args.val_data)
    images, labels = images[: args.val_size], labels[: args.val_size]
    for ckpt in args.ckpts:
        model = HydroDiffForImageGeneration.from_pretrained(ckpt)
        print(json.dumps({"ckpt": ckpt, **metrics(model, images, labels)}), flush=True)


if __name__ == "__main__":
    main()
