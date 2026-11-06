"""
Trains a text-to-image HydroDiff model in one pass over a streamed caption dataset: nothing is stored on disk, every
image is used once (one epoch), at its own size and without cropping (each side is scaled to the nearest multiple of
the patch size, at most half a patch).

    python test/train_stream.py --out runs/coco --text-encoder models/clip-vit-base-patch32-text --max-bytes 1e9

* Rows are read one parquet row group at a time with range requests, only the image and caption columns, and every
  byte downloaded is counted (metadata and validation images included). The stream stops before the row group that
  would take the download past `--max-bytes`, or at `--limit-images`.
* Images are batched by size (`--batch` per size; a buffer of `--buffer` images flushes its largest size when full,
  and everything left over is trained at the end), so no image is padded, cropped or dropped.
* Each image is trained with one of its captions, picked at random; the text encoder is frozen and runs on the fly.
* Validation: the first `--val-size` streamed images are held out (never trained on); fixed noise and noise levels,
  as in test/train.py.
* Running the same command again resumes from the last saved state (images still buffered at that moment are skipped).
"""

import argparse
import ast
import copy
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff import HydroDiffConfig, HydroDiffForImageGeneration  # noqa: E402
from hydrodiff.optimization_hydrodiff import build_optimizers, lr_factor, set_lr_factor  # noqa: E402
from hydrodiff.pipeline_hydrodiff import HydroDiffPipeline, pil_to_tensor  # noqa: E402


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def captions_of(row) -> list[str]:
    raw = row.get("sentences_raw", row.get("captions", row.get("caption", row.get("text"))))
    if isinstance(raw, str) and raw.startswith("["):
        raw = ast.literal_eval(raw)
    return [raw] if isinstance(raw, str) else list(raw)


class CountingFile(io.RawIOBase):
    """A read-only file that counts every byte fetched from the remote file (the whole download)."""

    def __init__(self, f, counter: list):
        self.f, self.counter = f, counter

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        return self.f.seek(offset, whence)

    def tell(self):
        return self.f.tell()

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, size=-1):
        data = self.f.read(size)
        self.counter[0] += len(data)
        return data

    def size(self):
        return self.f.size


COLUMNS = ("image", "sentences_raw")


def stream(name: str, split: str, budget: float, counter: list, skip_rows: int = 0, fs=None, root: str = "datasets"):
    """
    Yields the rows of a Hugging Face parquet dataset, one row group at a time, straight from the Hub with range
    requests (nothing is written to disk). Only the image and caption columns are fetched. `counter[0]` holds every
    byte downloaded (metadata included); a row group that would take it past `budget` is not downloaded.
    `fs` / `root`: another fsspec filesystem and folder (tests use a local one).
    """
    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    files = sorted(fs.glob(f"{root}/{name}/data/{split}-*.parquet"))
    seen = 0
    for path in files:
        handle = CountingFile(fs.open(path, "rb", cache_type="none"), counter)
        parquet = pq.ParquetFile(handle)  # unbuffered: only the byte ranges parquet asks for are fetched
        names = parquet.schema_arrow.names
        columns = [c for c in COLUMNS if c in names] or names
        for g in range(parquet.num_row_groups):
            meta = parquet.metadata.row_group(g)
            if seen + meta.num_rows <= skip_rows:
                seen += meta.num_rows
                continue
            cost = sum(meta.column(j).total_compressed_size for j in range(meta.num_columns)
                       if meta.column(j).path_in_schema.split(".")[0] in columns)
            if counter[0] + cost > budget:
                return
            table = parquet.read_row_group(g, columns=columns)
            for start in range(0, table.num_rows, 64):  # small slices: the row group is held once, not twice
                for row in table.slice(start, 64).to_pylist():
                    seen += 1
                    if seen > skip_rows:
                        yield row
            del table


def decode(row, multiple: int):
    """uint8 (3, h, w): buffered images take a quarter of the memory of float pixels."""
    data = row["image"]["bytes"]
    pixels = pil_to_tensor(Image.open(io.BytesIO(data)), multiple)
    return pixels.add(1).mul(127.5).round().to(torch.uint8), len(data)


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    return x.float().div(127.5).sub(1)


class TextEncoder:
    def __init__(self, path: str, max_length: int):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.model = CLIPTextModelWithProjection.from_pretrained(path).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, texts):
        tokens = self.tokenizer(texts, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        out = self.model(**tokens)
        return out.last_hidden_state.clone(), out.text_embeds.clone()


@torch.no_grad()
def validation_loss(model, val, seed=1234):
    """Mean velocity loss on fixed images/prompts/noise/levels; (uniform, mixed) as in test/train.py."""
    was_training = model.training
    model.eval()
    results = []
    for mixed in (False, True):
        g = torch.Generator().manual_seed(seed)
        total = 0.0
        for i, (x, text, pool) in enumerate(val):
            n = (x.shape[-2] // model.config.patch_size) * (x.shape[-1] // model.config.patch_size)
            noise = torch.randn(1, n, model.config.patch_dim, generator=g)
            t = torch.full((1, n), (i + 0.5) / len(val))
            if mixed:
                t = (1 - (1 - t) * (1 + torch.rand(1, n, generator=g))).clamp(0, 1)
            total += model(x[None], timesteps=t, noise=noise, text=text, text_pool=pool).denoise_loss.item()
        results.append(total / len(val))
    model.train(was_training)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Multimodal-Fatima/COCO_captions_train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-size", type=int, default=32)
    ap.add_argument("--max-bytes", type=float, default=1e9, help="bytes downloaded, validation included")
    ap.add_argument("--limit-images", type=int, default=0, help="stop after this many training images (0: no limit)")
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--preset", default="hydrodiff-s")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=2, help="images per step (2 beat 4 and 8 by 11-13%% at one epoch)")
    ap.add_argument("--buffer", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-pipeline", action="store_true", help="also save a ready-to-use pipeline (with text encoder)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    overrides = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    # measured on 960 COCO images: the JiT bottleneck embedding is 14% lower in validation loss, 6% smaller, as fast
    defaults = dict(image_size=512, patch_size=32, patch_embed_dim=128, num_classes=0, text_dim=512, modalities=["image"])
    config = HydroDiffConfig.from_preset(args.preset, **{**defaults, **overrides})
    model = HydroDiffForImageGeneration(config).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizers = build_optimizers(model, lr=args.lr, weight_decay=0.0, optimizer="muon", muon_lr=args.muon_lr)
    encoder = TextEncoder(args.text_encoder, config.text_max_length)
    multiple = config.patch_size * config.vae_scale_factor
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params / 1e6:.2f}M  config {json.dumps(overrides)}", flush=True)

    state_path = os.path.join(args.out, "state.pt")
    step, rows, images_done, elapsed0 = 0, 0, 0, 0.0
    downloaded = [0]  # every byte fetched from the Hub, validation images and metadata included
    val = []
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        for opt, s in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(s)
        step, rows, images_done, elapsed0 = (state[k] for k in ("step", "rows", "images", "elapsed"))
        downloaded[0], val = state["bytes"], state["val"]
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed at step {step}, row {rows}", flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    start = time.time() - elapsed0
    buckets: dict[tuple[int, int], list] = {}
    buffered, seen_loss = 0, []

    def planned_steps():
        if args.limit_images:
            total = args.limit_images
        else:
            per_row = downloaded[0] / max(1, rows)
            total = args.max_bytes / max(per_row, 1) - len(val)
        return max(step + 1, int(total / args.batch))

    def train_on(batch):
        nonlocal step, images_done
        x = to_model_range(torch.stack([b[0] for b in batch]))
        text, pool = encoder([b[1] for b in batch])
        flip = torch.rand(len(batch)) < 0.5  # horizontal flips (captions rarely depend on left/right)
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        set_lr_factor(optimizers, lr_factor(step, planned_steps(), args.warmup, "wsd"))
        out = model(x, text=text, text_pool=pool)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            decay = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        seen_loss.append(out.denoise_loss.item())
        step += 1
        images_done += len(batch)
        if step % args.eval_every == 0:
            report()
        if step % args.save_every == 0:
            state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizers=[o.state_dict() for o in optimizers],
                         step=step, rows=rows - buffered, images=images_done, bytes=downloaded[0], val=val,
                         elapsed=time.time() - start, rng=rng.getstate(), torch_rng=torch.get_rng_state())
            torch.save(state, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)

    def report(final=False):
        uniform, mixed = validation_loss(ema, val)
        raw_uniform, _ = validation_loss(model, val) if final else (None, None)
        minutes = (time.time() - start) / 60
        rec = dict(step=step, images=images_done, mb=round(downloaded[0] / 1e6, 1), minutes=round(minutes, 2),
                   train=float(np.mean(seen_loss)) if seen_loss else None, val_uniform=uniform, val_mixed=mixed,
                   imgs_per_sec=round(images_done / max(1e-9, minutes * 60), 3))
        if final:
            rec.update(final=True, raw_val_uniform=raw_uniform, params=params)
        seen_loss.clear()
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    exhausted = True
    for row in stream(args.dataset, args.split, args.max_bytes, downloaded, skip_rows=rows):
        rows += 1
        x, _ = decode(row, multiple)
        if len(val) < args.val_size:  # the first rows are held out for validation, never trained on
            text, pool = encoder([captions_of(row)[0]])
            val.append((to_model_range(x), text, pool))
            continue
        if args.limit_images and images_done + buffered >= args.limit_images:
            exhausted = False
            break
        bucket = buckets.setdefault(tuple(x.shape[-2:]), [])
        bucket.append((x, rng.choice(captions_of(row))))
        buffered += 1
        if len(bucket) == args.batch:
            buffered -= len(bucket)
            train_on(buckets.pop(tuple(x.shape[-2:])))
        elif buffered >= args.buffer:  # flush the largest partial size, so memory stays bounded
            key = max(buckets, key=lambda k: len(buckets[k]))
            batch = buckets.pop(key)
            buffered -= len(batch)
            train_on(batch)
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):  # the leftovers: every image is trained once
        batch = buckets.pop(key)
        buffered -= len(batch)
        train_on(batch)
    report(final=True)
    print(json.dumps(dict(stopped="byte budget or end of data" if exhausted else "image limit",
                          images=images_done, downloaded_gb=round(downloaded[0] / 1e9, 4))), flush=True)
    ema.save_pretrained(os.path.join(args.out, "ema"))
    model.save_pretrained(os.path.join(args.out, "raw"))
    if args.save_pipeline:
        pipe = HydroDiffPipeline(ema, encoder.model, encoder.tokenizer)
        pipe.save_pretrained(os.path.join(args.out, "pipeline"), half=True)


if __name__ == "__main__":
    main()
