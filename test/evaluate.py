"""
Samples from a checkpoint and reports FID against CIFAR-10 train images, sampling time and compute.

    python test/evaluate.py --ckpt runs/base/ema --n 5000 --gen steps=32 adaptive=false
    python test/evaluate.py --ckpt runs/base/ema --n 5000 --gen steps=32 adaptive=true max_rate=2

`--gen key=value` is passed to `HydroDiffForImageGeneration.generate` (`guide=<checkpoint>` loads a weaker model for
autoguidance). FID uses the pytorch-fid InceptionV3 (2048-d
pool features). With fewer than 50k samples FID is biased upwards, so only compare numbers measured with the same `--n`.
`compute` is token evaluations relative to a dense sampler with the same steps. `--compare` also runs a dense sampler
with the same seeds and reports how far the images move (PSNR), which isolates the error of a fast sampler.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff import HydroDiffForImageGeneration  # noqa: E402


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def inception():
    from pytorch_fid.inception import InceptionV3

    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()


@torch.no_grad()
def features(net, images01: torch.Tensor, batch=200) -> np.ndarray:
    out = [net(images01[i : i + batch])[0].flatten(1) for i in range(0, len(images01), batch)]
    return torch.cat(out).double().numpy()


def frechet(f1: np.ndarray, f2_stats: tuple[np.ndarray, np.ndarray]) -> float:
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2_stats
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def reference_stats(path: str, data: str, n: int, net) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(path):
        ref = np.load(path)
        return ref["mu"], ref["sigma"]
    x = np.load(data)["x"]
    idx = np.random.default_rng(0).permutation(len(x))[:n]
    images = torch.from_numpy(x[idx]).permute(0, 3, 1, 2).float() / 255
    f = features(net, images)
    mu, sigma = f.mean(0), np.cov(f, rowvar=False)
    np.savez(path, mu=mu, sigma=sigma)
    return mu, sigma


@torch.no_grad()
def sample(model, n, batch, seed, gen):
    labels = torch.arange(n) % model.config.num_classes if model.config.num_classes else None
    images, compute, seconds = [], [], 0.0
    for i in range(0, n, batch):
        g = torch.Generator().manual_seed(seed + i)
        lab = None if labels is None else labels[i : i + batch]
        start = time.perf_counter()
        out = model.generate(labels=lab, batch_size=min(batch, n - i), generator=g, **gen)
        seconds += time.perf_counter() - start
        images.append(out.images)
        compute.append(out.compute)
    return torch.cat(images), float(np.mean(compute)), seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true", help="also report PSNR against a dense sampler, same seeds")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default="/home/user/data/cifar10_train.npz")
    ap.add_argument("--ref", default="/home/user/data/fid_ref_train10k.npz", help="reference statistics, computed once and cached: delete the file after changing --data or --ref-size")
    ap.add_argument("--ref-size", type=int, default=10000)
    ap.add_argument("--save-grid", default=None, help="write the first 100 samples as a PNG grid")
    ap.add_argument("--results", default=None, help="append the result as a JSON line to this file")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-fid", action="store_true", help="only time the sampler")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    gen = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.gen)}
    model = HydroDiffForImageGeneration.from_pretrained(args.ckpt).eval()
    sampler_args = dict(gen)
    if "guide" in sampler_args:  # autoguidance: guide=<checkpoint of a weaker model>
        sampler_args["guide_model"] = HydroDiffForImageGeneration.from_pretrained(sampler_args.pop("guide")).eval()
    images, compute, seconds = sample(model, args.n, args.batch, args.seed, sampler_args)
    result = dict(tag=args.tag, ckpt=args.ckpt, n=args.n, gen=gen, compute=round(compute, 4),
                  ms_per_image=round(1000 * seconds / args.n, 3))
    if args.compare:  # same steps, solver and guidance; every token computed in every step
        dense_gen = {k: v for k, v in gen.items() if k in ("steps", "solver", "guidance_scale")}
        dense_gen["adaptive"] = False
        dense, _, dense_seconds = sample(model, args.n, args.batch, args.seed, dense_gen)
        mse = (images - dense).pow(2).flatten(1).mean(1)
        result["psnr_vs_dense"] = round(float((10 * torch.log10(4 / mse.clamp_min(1e-10))).mean()), 3)
        result["dense_ms_per_image"] = round(1000 * dense_seconds / args.n, 3)
    if args.save_grid:
        from torchvision.utils import save_image

        save_image((images[:100] + 1) / 2, args.save_grid, nrow=10)
    if not args.no_fid:
        net = inception()
        ref = reference_stats(args.ref, args.data, args.ref_size, net)
        images = ((images + 1) * 127.5).round() / 255  # quantised like the reference images
        result["fid"] = round(frechet(features(net, images), ref), 3)
    print(json.dumps(result), flush=True)
    if args.results:
        with open(args.results, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
