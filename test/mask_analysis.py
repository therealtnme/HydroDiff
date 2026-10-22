"""
What the noise mask sees.

    python test/mask_analysis.py --ckpt runs/base/ema --figure test/figures/noise_mask.png

1. Correlation: on test images at fixed noise levels, how does each token's predicted error (the source of the noise
   mask) rank against how much detail the clean patch has (pixel standard deviation and high-pass energy)? Spearman
   correlation per image, averaged.
2. Figure: generated images above the step at which each of their tokens finished (dark = the earliest finishing step
   in the figure, bright = the last step).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff import HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.modeling_hydrodiff import patchify  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_images, to_model_range  # noqa: E402


@torch.no_grad()
def correlation(model, images, labels, levels=(0.3, 0.6, 0.9), batch=250, seed=0):
    cfg = model.config
    g = torch.Generator().manual_seed(seed)
    x = to_model_range(images)
    x0 = patchify(x, cfg.patch_size)
    detail = x0.std(-1)  # per-token pixel standard deviation
    blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    high_pass = patchify(x - blurred, cfg.patch_size).pow(2).mean(-1)  # per-token high-frequency energy
    out = {}
    for t in levels:
        log_error = []
        for i in range(0, len(x), batch):
            b = x0[i : i + batch]
            noise = torch.randn(b.shape, generator=g)
            ts = torch.full(b.shape[:2], t)
            x_t = (1 - t) * b + t * noise
            log_error.append(model.model(x_t, ts, labels[i : i + batch]).log_error)
        log_error = torch.cat(log_error)
        for name, measure in (("std", detail), ("high_pass", high_pass)):
            rho = [spearmanr(log_error[j].numpy(), measure[j].numpy()).statistic for j in range(len(x))]
            out[f"rho_{name}@t={t}"] = round(float(np.nanmean(rho)), 3)
    return out


def ramp(values: np.ndarray) -> np.ndarray:
    """Dark blue (0) to yellow (1)."""
    low, high = np.array([30, 30, 90]), np.array([250, 230, 60])
    return (low + (high - low) * values[..., None]).astype(np.uint8)


@torch.no_grad()
def figure(model, path, count=10, steps=32, seed=0, scale=4):
    labels = torch.arange(count) % max(model.config.num_classes, 1)
    out = model.generate(labels=labels, steps=steps, solver="dpm2m", return_masks=True,
                         generator=torch.Generator().manual_seed(seed))
    side = model.config.patches_per_side
    finished = (out.noise_masks[1:] == 0).float().argmax(0)  # (batch, tokens): first step at which t = 0
    images = ((out.images + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    size = model.config.image_size * scale
    canvas = Image.new("RGB", (count * (size + 4), 2 * size + 4), "white")
    for i in range(count):
        canvas.paste(Image.fromarray(images[i]).resize((size, size), Image.NEAREST), (i * (size + 4), 0))
        span = (finished - finished.min()).float() / max(int(finished.max() - finished.min()), 1)
        heat = ramp(span[i].view(side, side).numpy())
        canvas.paste(Image.fromarray(heat).resize((size, size), Image.NEAREST), (i * (size + 4), size + 4))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)
    return {"compute": round(out.compute, 3), "mean_finish_step": round(float(finished.float().mean()) + 1, 2),
            "first_finish_step": int(finished.min()) + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model = HydroDiffForImageGeneration.from_pretrained(args.ckpt).eval()
    images, labels = load_images("/home/user/data/cifar10_test.npz")
    result = {"ckpt": args.ckpt, **correlation(model, images[: args.n], labels[: args.n])}
    if args.figure:
        result.update(figure(model, args.figure))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
