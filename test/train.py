"""
Trains a HydroDiff model on CIFAR-10 (class-conditional, 32x32) on CPU or GPU.

    python test/train.py --preset hydrodiff-s --minutes 60 --out runs/base
    python test/train.py --preset hydrodiff-s --set hetero_prob=0 --minutes 60 --out runs/no_hetero

`--set key=value` overrides any config field. The data file is a .npz with `x` (N, 32, 32, 3) uint8 and `y` (N,) labels
(test/prepare_cifar10.py writes it). Validation loss is measured on fixed test images, noise and noise levels, both with
one noise level per image ("uniform") and with per-token levels at speeds in [1, 2] ("mixed"); test/validate.py
measures saved checkpoints the same way, plus the clean-image error per noise band.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff import HydroDiffConfig, HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.optimization_hydrodiff import build_optimizers, lr_factor, set_lr_factor  # noqa: E402


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def load_images(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    return torch.from_numpy(data["x"]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(data["y"])


def to_model_range(images_uint8: torch.Tensor) -> torch.Tensor:
    return images_uint8.float() / 127.5 - 1


@torch.no_grad()
def validation_loss(model, images, labels, seed=1234, batch=250):
    """Mean velocity loss on fixed images/noise/levels; returns (uniform, mixed)."""
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    results = []
    for mixed in (False, True):
        total, count = 0.0, 0
        for i in range(0, len(images), batch):
            x = to_model_range(images[i : i + batch])
            b = x.shape[0]
            noise = torch.randn(b, model.config.num_patches, model.config.patch_dim, generator=g)
            t = ((torch.arange(b) + 0.5) / b)[:, None].expand(b, model.config.num_patches)
            if mixed:  # fixed definition (not the model's own hetero_prob), so every config is measured alike
                rate = 1 + torch.rand(b, model.config.num_patches, generator=g)
                t = (1 - (1 - t) * rate).clamp(0, 1)
            out = model(x, labels[i : i + batch], timesteps=t, noise=noise)
            total += out.denoise_loss.item() * b
            count += b
        results.append(total / count)
    model.train(was_training)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="hydrodiff-s")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--data", default="/home/user/data/cifar10_train.npz")
    ap.add_argument("--val-data", default="/home/user/data/cifar10_test.npz")
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=0, help="total steps (0: run for --minutes)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--optimizer", default="muon")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--extra-ema", type=float, nargs="*", default=[], help="more EMA decays, saved as ema_<decay>")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=250, help="full training state for resuming after an interruption")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint dir to continue from (weights only)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: parse_value(v) for k, v in overrides.items()}
    if args.resume:
        if overrides:
            raise ValueError("--set cannot change the configuration of a resumed checkpoint")
        model = HydroDiffForImageGeneration.from_pretrained(args.resume)
    else:
        model = HydroDiffForImageGeneration(HydroDiffConfig.from_preset(args.preset, **overrides))
    model.train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    emas = {args.ema: ema, **{d: copy.deepcopy(ema) for d in args.extra_ema}}
    images, labels = load_images(args.data)
    val_images, val_labels = load_images(args.val_data)
    val_images, val_labels = val_images[: args.val_size], val_labels[: args.val_size]
    optimizers = build_optimizers(
        model, lr=args.lr, weight_decay=args.weight_decay, optimizer=args.optimizer, muon_lr=args.muon_lr
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params / 1e6:.2f}M  config {json.dumps(overrides)}", flush=True)

    total_steps = args.steps
    budget = args.minutes * 60
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    start, step, seen_loss = time.time(), 0, []
    g = torch.Generator().manual_seed(args.seed)
    # Re-running the same command continues from the last saved state (weights, EMA, optimizers, step and RNG)
    state_path = os.path.join(args.out, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        for d, extra in state.get("extra_ema", {}).items():
            if d in emas:
                emas[d].load_state_dict(extra)
        for opt, opt_state in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        step = state["step"]
        g.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start -= state["elapsed"]
        print(f"resumed at step {step}", flush=True)
    while True:
        elapsed = time.time() - start
        if total_steps and step >= total_steps:
            break
        if not total_steps and elapsed >= budget:
            break
        if total_steps:
            planned = total_steps
        else:  # time budget: estimate the total from the speed so far (decay starts at 80% of the budget)
            planned = int(budget / (elapsed / step)) if step >= 20 else 10**6
        set_lr_factor(optimizers, lr_factor(step, planned, args.warmup, "wsd"))
        idx = torch.randint(0, len(images), (args.batch,), generator=g)
        x = to_model_range(images[idx])
        flip = torch.rand(args.batch, generator=g) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        out = model(x, labels[idx])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            warm = min(args.ema, (1 + step) / (10 + step))
            for target, average in emas.items():
                # every EMA follows the main one's warmup with its half-life scaled, so each reaches its own decay
                decay = 1 - (1 - warm) * (1 - target) / (1 - args.ema)
                for pe, pm in zip(average.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
                for be, bm in zip(average.buffers(), model.buffers()):
                    be.copy_(bm)
        seen_loss.append(out.denoise_loss.item())
        step += 1
        if step % args.eval_every == 0:
            uniform, mixed = validation_loss(ema, val_images, val_labels)
            rec = dict(step=step, minutes=(time.time() - start) / 60, train=float(np.mean(seen_loss)),
                       val_uniform=uniform, val_mixed=mixed, lr_factor=optimizers[0].param_groups[0]["lr"] / optimizers[0].param_groups[0]["base_lr"])
            seen_loss = []
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            save_emas(emas, ema, args.out)
        if step % args.save_every == 0:
            state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                         extra_ema={d: e.state_dict() for d, e in emas.items() if e is not ema},
                         step=step, generator=g.get_state(), torch_rng=torch.get_rng_state(), elapsed=time.time() - start)
            torch.save(state, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    uniform, mixed = validation_loss(ema, val_images, val_labels)
    rec = dict(step=step, minutes=(time.time() - start) / 60, final=True, val_uniform=uniform, val_mixed=mixed,
               params=params, imgs_per_sec=step * args.batch / (time.time() - start))
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    save_emas(emas, ema, args.out)


def save_emas(emas, ema, out):
    for decay, average in emas.items():
        average.save_pretrained(os.path.join(out, "ema" if average is ema else f"ema_{decay}"))


if __name__ == "__main__":
    main()
