#!/usr/bin/env python3
"""How does one forward of the prompt rewriter scale with sequence length?

Motivation: on the GPU the fp8 rewriter is 10.24 GiB resident of ~13.7 GiB free, and a
prefill of the full 2426-token system prompt never completed - `dmesg` shows the dxg
allocations failing with -12 (ENOMEM), so some activation scales with the square of the
prompt length. This measures the curve at forward level (no sampling, no cache) so the
choice between "chunk the prefill", "shrink the prompt" and "use the CPU" is made from
numbers instead of guesswork.

    $QIP_VENV/bin/python scripts/bench_pe_seqlen.py --lengths 32 128 512 1024
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_rewrite  # noqa: E402

GIB = 2 ** 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("QIP_PE_MODEL", pe_rewrite.DEFAULT_MODEL))
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    model, tokenizer, _ = pe_rewrite.load_rewriter(args.model, device=args.device)
    dev = torch.device(getattr(model, "device", "cpu"))
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[bench] vram free {free/GIB:.2f} / {total/GIB:.2f} GiB after load", flush=True)

    for n in args.lengths:
        ids = torch.randint(100, 200000, (1, n), device=dev)
        mask = torch.ones_like(ids)
        try:
            with torch.no_grad():
                t0 = time.perf_counter()
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            logits = out.logits
            print(f"[bench] len {n:5d}: {dt*1000:9.1f} ms   logits {tuple(logits.shape)} "
                  f"{logits.dtype} = {logits.numel()*logits.element_size()/GIB:.2f} GiB",
                  flush=True)
        except Exception as e:  # OOM here is a data point, not a crash
            print(f"[bench] len {n:5d}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        else:
            del out, logits
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            print(f"[bench]        vram free {free/GIB:.2f} GiB", flush=True)
        del ids, mask
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
