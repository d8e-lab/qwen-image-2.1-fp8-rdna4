#!/usr/bin/env python3
"""Benchmark a real Qwen-Image-2.1 transformer forward pass on the GPU (or CPU).

Purpose: the fp8 weights moved to the GPU fine, but the first GPU forward took 250 s.
The cause was a layout path: a linear whose weight is a *transposed view* falls into a
degenerate hipBLASLt path on this ROCm build (0.10 TFLOP/s vs 84 TFLOP/s for a
contiguous (in, out) weight). `Fp8Weight` now stores/materialises the weight transposed
and computes `x @ Wt`, so this script measures whether that actually fixed the model.

Writes to logs/gpu_fwd.log with timestamps. Use --device cpu to compare.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import fp8_memory_summary, quantize_linears_

GIB = 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=64, help="latent grid side (64 -> 4096 tokens)")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "gpu_fwd.log")
    logfile = os.path.normpath(logfile)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    LOG = open(logfile, "w")

    def p(*a):
        print(datetime.datetime.now().strftime("%H:%M:%S"), *a, file=LOG, flush=True)

    from diffusers import QwenImage21Transformer2DModel

    model_dir = os.environ.get(
        "QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
    )
    p(f"device={args.device} grid={args.grid} ({args.grid**2} tokens)")
    p("loading transformer ...")
    tf = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(model_dir, "transformer"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    quantize_linears_(tf, min_params=1 << 16)
    p("fp8: %.3f GiB -> moving to %s" % (fp8_memory_summary(tf)["total_GiB"], args.device))
    t0 = time.perf_counter()
    tf = tf.to(args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    p("moved in %.1fs" % (time.perf_counter() - t0))

    grid = args.grid
    tokens = grid * grid
    text_tokens = 32
    dev = args.device
    x = torch.randn(1, tokens, tf.config.in_channels, dtype=torch.bfloat16, device=dev)
    ctx = torch.randn(1, text_tokens, tf.config.context_in_dim, dtype=torch.bfloat16, device=dev)
    t = torch.tensor([0.5], dtype=torch.bfloat16, device=dev)
    shapes = [[(1, grid, grid)]]
    mask = torch.cat([
        torch.zeros(1, text_tokens, dtype=torch.bool),
        torch.ones(1, tokens // 4, dtype=torch.bool),
    ], dim=1).to(dev)

    def step():
        with torch.no_grad():
            return tf(x, encoder_hidden_states=ctx, timestep=t,
                      img_shapes=shapes, img_mask=mask, return_dict=False)[0]

    p("warmup forward ...")
    t0 = time.perf_counter()
    o = step()
    if dev == "cuda":
        torch.cuda.synchronize()
    p("  first forward: %.2f s  finite=%s" % (time.perf_counter() - t0, bool(torch.isfinite(o).all())))

    for i in range(args.iters):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = step()
        if dev == "cuda":
            torch.cuda.synchronize()
        extra = ""
        if dev == "cuda":
            extra = "  free %.2f GiB" % (torch.cuda.mem_get_info()[0] / GIB)
        p("  forward %d: %.3f s%s" % (i + 1, time.perf_counter() - t0, extra))

    p("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
