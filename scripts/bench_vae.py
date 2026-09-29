#!/usr/bin/env python3
"""Measure VAE decode cost in isolation, and how it responds to tiling / dtype / slicing.

The VAE decode is a surprisingly large share of a run (the transformer is only 0.63 GiB of
the budget but the decoder has base_dim 144 with 2x upsampling stages). This script isolates
it so the settings can be chosen from measurements rather than guessed.

Usage:
    python scripts/bench_vae.py --height 1024 --width 1024
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import torch

GIB = 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logfile = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "bench_vae.log")
    )
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    LOG = open(logfile, "w")

    def p(*a):
        print(datetime.datetime.now().strftime("%H:%M:%S"), *a, file=LOG, flush=True)

    from diffusers import AutoencoderKLQwenImage21

    model_dir = os.environ.get(
        "QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
    )
    z = 64                      # latent channels
    lat_h = args.height // 16
    lat_w = args.width // 16
    p(f"target {args.width}x{args.height} -> latent {lat_w}x{lat_h} x{z} channels")

    for dtype_name, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        for tiling in (False, True):
            vae = AutoencoderKLQwenImage21.from_pretrained(
                os.path.join(model_dir, "vae"), torch_dtype=dtype, low_cpu_mem_usage=True
            ).eval()
            if tiling:
                vae.enable_tiling()
            vae.to(args.device)
            lat = torch.randn(1, z, 1, lat_h, lat_w, dtype=dtype, device=args.device)

            def run():
                with torch.no_grad():
                    return vae.decode(lat, return_dict=False)[0]

            with torch.no_grad():
                try:
                    o = run()                      # warmup
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(args.iters):
                        o = run()
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) / args.iters
                    extra = ""
                    if args.device == "cuda":
                        extra = "  peak %.2f GiB" % (torch.cuda.max_memory_allocated() / GIB)
                    p(f"  dtype={dtype_name:9s} tiling={str(tiling):5s}  {dt*1000:8.1f} ms  "
                      f"out={tuple(o.shape)}{extra}")
                except Exception as e:
                    p(f"  dtype={dtype_name:9s} tiling={str(tiling):5s}  FAILED: "
                      f"{type(e).__name__}: {str(e)[:90]}")
            del vae, lat, o
            if args.device == "cuda":
                torch.cuda.empty_cache()
    p("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
