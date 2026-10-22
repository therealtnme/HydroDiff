"""
Runs a list of sampler settings through test/evaluate.py in parallel and prints a table.

    python test/sampler_study.py --ckpt runs/base/ema --n 5000 --workers 4 --results runs/sampler.jsonl \
        "dense-euler-32: steps=32 adaptive=false" \
        "adaptive-32: steps=32 adaptive=true max_rate=2"

Each setting is `name: key=value ...` (passed to `generate`; `seed=N` sets the sampling seed instead). Settings already in
the results file (same name, checkpoint and n) are skipped, so a study can be extended and re-run.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(spec: str, args) -> dict | None:
    name, _, gen = spec.partition(":")
    seed = [kv.split("=", 1)[1] for kv in gen.split() if kv.startswith("seed=")]
    gen = " ".join(kv for kv in gen.split() if not kv.startswith("seed="))
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"),
        "--ckpt", args.ckpt, "--n", str(args.n), "--threads", str(args.threads), "--tag", name.strip(),
        "--results", args.results, "--seed", seed[0] if seed else "0", "--gen", *gen.split(),
    ]
    if args.compare:
        cmd.append("--compare")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        print(f"[{name}] failed:\n{out.stderr[-2000:]}", flush=True)
        return None
    result = json.loads(out.stdout.strip().splitlines()[-1])
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--results", required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.results):
        for line in open(args.results):
            r = json.loads(line)
            done.add((r["tag"], r["ckpt"], r["n"]))
    todo = [s for s in args.specs if (s.partition(":")[0].strip(), args.ckpt, args.n) not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda s: run(s, args), todo))

    rows = [json.loads(line) for line in open(args.results)]
    rows = [r for r in rows if r["ckpt"] == args.ckpt and r["n"] == args.n]
    print(f"\n| setting | FID | compute | ms / image |{' PSNR vs dense |' if args.compare else ''}")
    print(f"|---|---|---|---|{'---|' if args.compare else ''}")
    for r in rows:
        extra = f" {r.get('psnr_vs_dense', '')} |" if args.compare else ""
        print(f"| {r['tag']} | {r.get('fid', '-')} | {r['compute']} | {r['ms_per_image']} |{extra}")


if __name__ == "__main__":
    main()
