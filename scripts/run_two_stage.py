#!/usr/bin/env python3
"""End-to-end: a short request in any language -> Qwen-Image-2.1 render on this box.

Stage 1 runs Qwen-Image-2.1-PE-T2I on CPU and turns e.g. "一只在雨中弹吉他的柯基" into a
detailed English prompt plus a recommended aspect ratio. Stage 2 hands that prompt to
`scripts/run_qwen_image.py` (the fp8 image model) and writes a PNG.

Why two processes
=================
The two models cannot share this machine at the same time, and the constraint is RAM as
much as VRAM:

    rewriter (BF16)                18.80 GiB   > 15.82 GiB VRAM, so CPU only
    image transformer (bf16)       13.25 GiB   hosted while quantising to fp8
    image text encoder (bf16)      16.33 GiB   hosted while encoding the prompt

The VM has ~21 GiB available. Running both stages in one process would mean holding the
rewriter while the image model's staged loading begins; letting the rewriter's process
exit is the only unambiguous way to return its memory, so stage 2 is a subprocess.

Resolution
==========
The card's advised render sizes are 2048x2048 and larger - 4x the pixels of a 1024x1024
run, i.e. roughly 4x the attention KV cache and activation peak that already sits at
~10.3 GiB of ~14 GiB free on this box. So the aspect ratio is honoured but the *area* is
budgeted: the longest side is --max-side (default 1024) and the short side follows the
ratio, rounded to a multiple of 16.

Usage
=====
    source scripts/env.sh
    $QIP_VENV/bin/python scripts/run_two_stage.py \
        --prompt "一只在雨中弹吉他的柯基" --output corgi.png
    $QIP_VENV/bin/python scripts/run_two_stage.py --prompt "..." --dry-run
    $QIP_VENV/bin/python scripts/run_two_stage.py --prompt "..." -- --steps 30 --vae-tiling
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe_rewrite  # noqa: E402

RUNNER = os.path.join(HERE, "run_qwen_image.py")
GIB = 2 ** 30


def parse_ratio(ratio: str) -> tuple[int, int]:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", str(ratio))
    if not m:
        return 1, 1
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return 1, 1
    return a, b


def fit_size(ratio: str, max_side: int = 1024, multiple: int = 16) -> tuple[int, int]:
    """Render size for an aspect ratio under a pixel-area budget.

    Longest side is pinned to ``max_side`` and the short side follows, both rounded to a
    multiple of ``multiple`` (the VAE works on a 16px grid; odd sizes cost a pad and a
    crop for nothing). 1:1  -> 1024x1024, 16:9 -> 1024x576, 2:3 -> 688x1024.
    """
    a, b = parse_ratio(ratio)
    if a >= b:
        w = max_side
        h = max_side * b / a
    else:
        h = max_side
        w = max_side * a / b
    rw = max(multiple, int(round(w / multiple)) * multiple)
    rh = max(multiple, int(round(h / multiple)) * multiple)
    return rw, rh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="short request, any language")
    ap.add_argument("--output", default="two_stage.png")
    ap.add_argument("--rewrite-model", default=os.environ.get("QIP_PE_MODEL", pe_rewrite.DEFAULT_MODEL))
    ap.add_argument("--no-rewrite", action="store_true",
                    help="skip stage 1 and pass --prompt straight to the image model")
    ap.add_argument("--max-new-tokens", type=int, default=pe_rewrite.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--rw-json", default=None, help="write {rewritten_prompt, wh_ratio} here")
    ap.add_argument("--reuse-rw", default=None,
                    help="reuse a previously written --rw-json instead of running stage 1 "
                         "(a rewrite costs ~35 min on CPU; re-rendering should not)")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="longest side of the render (default 1024; the card suggests 2048+)")
    ap.add_argument("--width", type=int, default=0, help="override the derived width")
    ap.add_argument("--height", type=int, default=0, help="override the derived height")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true", help="do stage 1 and print the plan only")
    ap.add_argument("--rewriter-only", action="store_true", help="stage 1 only")
    args, extra = ap.parse_known_args()

    t_all = time.perf_counter()
    rewritten = args.prompt
    ratio = "1:1"

    if args.no_rewrite:
        print("[1/2] rewrite skipped (--no-rewrite)", flush=True)
    elif args.reuse_rw:
        with open(args.reuse_rw, encoding="utf-8") as fh:
            cached = json.load(fh)
        rewritten, ratio = cached["rewritten_prompt"], cached.get("wh_ratio", "1:1")
        print(f"[1/2] reused {args.reuse_rw}")
        print(f"[1/2] ratio     : {ratio}")
        print(f"[1/2] rewritten :\n{rewritten}\n")
    else:
        print("=" * 78)
        print("[1/2] prompt rewrite (Qwen-Image-2.1-PE-T2I, CPU)")
        print("=" * 78)
        model, tokenizer, system_prompt = pe_rewrite.load_rewriter(
            args.rewrite_model, require_gib=20.0)
        result = pe_rewrite.rewrite(
            model, tokenizer, system_prompt, args.prompt,
            max_new_tokens=args.max_new_tokens, thinking=args.thinking,
        )
        # Free before anything else is imported/loaded, and report the RSS drop.
        del model, tokenizer
        import gc
        gc.collect()
        rewritten, ratio = result["rewritten_prompt"], result["wh_ratio"]
        print(f"\n[1/2] request   : {args.prompt}")
        print(f"[1/2] ratio     : {ratio}")
        print(f"[1/2] rewritten :\n{rewritten}\n")
        print(f"[1/2] {result['stats']}")
        if args.rw_json:
            with open(args.rw_json, "w", encoding="utf-8") as fh:
                json.dump({"rewritten_prompt": rewritten, "wh_ratio": ratio,
                           "request": args.prompt, "stats": result["stats"]},
                          fh, ensure_ascii=False, indent=2)
            print(f"[1/2] wrote {args.rw_json}")

    if args.rewriter_only:
        return 0

    w, h = fit_size(ratio, max_side=args.max_side)
    if args.width:
        w = args.width
    if args.height:
        h = args.height
    print(f"[2/2] render {w}x{h} ({ratio}, {w*h/1e6:.2f} MPix) through run_qwen_image.py")

    cmd = [sys.executable, "-u", RUNNER,
           "--prompt", rewritten,
           "--width", str(w), "--height", str(h),
           "--steps", str(args.steps), "--guidance", str(args.guidance),
           "--seed", str(args.seed), "--output", args.output,
           "--device", args.device]
    cmd += [e for e in extra if e != "--"]
    print("      " + " ".join(cmd[:6]) + " ... " + f"--output {args.output}", flush=True)
    if args.dry_run:
        print("\n--dry-run: not running stage 2")
        return 0

    rc = subprocess.run(cmd).returncode
    print(f"\n[2/2] exit {rc} after {time.perf_counter()-t_all:.1f}s total")
    if rc == 0 and os.path.isfile(args.output):
        print(f"[2/2] wrote {args.output} "
              f"({os.path.getsize(args.output)/1024:.0f} KiB)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
