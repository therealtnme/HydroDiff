"""
Measures how well a text-to-image HydroDiff pipeline follows its prompts, with a separate CLIP model as the judge:

    python test/evaluate_text.py --pipeline runs/coco/pipeline --clip models/clip-vit-base-patch32 --n 32

For `--n` held-out captions (one per image of `--dataset`, generated at that image's own size) it reports
(The evaluation photos come from the COCO test split, which training never reads, and are cached in `--cache`.)
* clip_score: mean 100 * cosine(image, its prompt) (higher = closer to the prompt; real photos give the ceiling),
* top1 / top5: how often the image's own prompt is the best (top 5) of all `--n` prompts (chance: 1/n, 5/n),
* seconds per image and the sampler's compute (relative to a dense sampler with the same steps),
and writes a grid of the generated images (with the real photos for comparison) to `--grid`.
`--gen key=value` passes sampler options (`steps=16 reuse=0.25 adaptive=false guidance_scale=1` ...).
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydrodiff.pipeline_hydrodiff import HydroDiffPipeline, pil_to_tensor, tensor_to_pil  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stream import captions_of, parse_value, stream  # noqa: E402


@torch.inference_mode()
def clip_scores(clip, processor, images, texts):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    out = clip(**inputs)
    img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
    sim = img @ txt.T  # (images, texts)
    ranks = (sim > sim.diag()[:, None]).sum(1)
    return dict(clip_score=round(100 * sim.diag().mean().item(), 2), top1=round((ranks == 0).float().mean().item(), 3),
                top5=round((ranks < 5).float().mean().item(), 3))


def load_eval_set(args):
    """`--n` held-out COCO test photos and their first captions, downloaded once into `--cache` (evaluation only)."""
    index = os.path.join(args.cache, "captions.json")
    if not os.path.exists(index):
        os.makedirs(args.cache, exist_ok=True)
        counter, texts = [0], []
        for i, row in enumerate(stream(args.dataset, args.split, float("inf"), counter)):
            if len(texts) >= args.skip + args.n:
                break
            if i < args.skip:
                continue
            with open(os.path.join(args.cache, f"{len(texts):04d}.jpg"), "wb") as f:
                f.write(row["image"]["bytes"])
            texts.append(captions_of(row)[0])
        json.dump(texts, open(index, "w"))
    texts = json.load(open(index))[: args.n]
    reals = [Image.open(os.path.join(args.cache, f"{i:04d}.jpg")).convert("RGB") for i in range(len(texts))]
    return reals, texts


def grid(pairs, path, cell=192):
    cols = 8
    rows = (len(pairs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (2 * cell + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (gen, real, text) in enumerate(pairs):
        x, y = (i % cols) * cell, (i // cols) * (2 * cell + 28)
        for j, im in enumerate((gen, real)):
            im = im.copy()
            im.thumbnail((cell, cell))
            canvas.paste(im, (x, y + j * cell))
        draw.text((x + 2, y + 2 * cell + 2), text[:34], fill="black")
        draw.text((x + 2, y + 2 * cell + 14), text[34:68], fill="black")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=None, help="a saved pipeline folder")
    ap.add_argument("--transformer", default=None, help="transformer weights (with --text-encoder, or inside --pipeline)")
    ap.add_argument("--text-encoder", default=None, help="CLIP text encoder + tokenizer folder (without --pipeline)")
    ap.add_argument("--clip", required=True, help="full CLIP model (vision + text) used as the judge")
    ap.add_argument("--dataset", default="Multimodal-Fatima/COCO_captions_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cache", default="eval_coco_test", help="where the evaluation photos are kept")
    ap.add_argument("--max-side", type=int, default=0, help="generate at most this long a side (0: native size)")
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grid", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import CLIPModel, CLIPProcessor

    from hydrodiff import HydroDiffForImageGeneration

    if args.pipeline:
        pipe = HydroDiffPipeline.from_pretrained(args.pipeline)
        if args.transformer:
            pipe.transformer = HydroDiffForImageGeneration.from_pretrained(args.transformer).eval()
    else:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        pipe = HydroDiffPipeline(HydroDiffForImageGeneration.from_pretrained(args.transformer),
                                 CLIPTextModelWithProjection.from_pretrained(args.text_encoder),
                                 CLIPTokenizer.from_pretrained(args.text_encoder))
    clip, processor = CLIPModel.from_pretrained(args.clip).eval(), CLIPProcessor.from_pretrained(args.clip)
    gen = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.gen)}
    steps = gen.pop("steps", 16)

    reals, texts = load_eval_set(args)
    images, compute, start = [], [], time.time()
    g = torch.Generator().manual_seed(args.seed)
    for real, text in zip(reals, texts):
        w, h = real.size
        if args.max_side and max(w, h) > args.max_side:
            w, h = w * args.max_side // max(w, h), h * args.max_side // max(w, h)
        out = pipe(text, height=h, width=w, num_inference_steps=steps, generator=g, **gen)
        images.append(out.images[0])
        compute.append(out.compute)
    seconds = (time.time() - start) / len(images)
    result = dict(generated=clip_scores(clip, processor, images, texts), real=clip_scores(clip, processor, reals, texts),
                  seconds_per_image=round(seconds, 2), compute=round(sum(compute) / len(compute), 3), steps=steps, gen=gen,
                  n=len(images))
    # a no-information baseline: the mean colour of each real photo, so a score above it means more than colour
    flat = [Image.new("RGB", r.size, tuple(int(c) for c in pil_to_tensor(r, 1).add(1).mul(127.5).mean((1, 2)))) for r in reals]
    result["mean_colour_baseline"] = clip_scores(clip, processor, flat, texts)
    print(json.dumps(result), flush=True)
    if args.grid:
        grid(list(zip(images, reals, texts)), args.grid)


if __name__ == "__main__":
    main()
