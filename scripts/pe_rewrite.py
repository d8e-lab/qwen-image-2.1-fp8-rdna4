#!/usr/bin/env python3
"""Rewrite a short image request into the long English prompt Qwen-Image-2.1 wants.

What this is
============
`Qwen/Qwen-Image-2.1-PE-T2I` is **not** an image model. It is the prompt planner that
ships next to Qwen-Image-2.1: a fine-tuned Qwen3.5-VL 9B (9.41B params, BF16 = 18.8 GiB
of weights) that turns a brief request in any language into

    {"rewritten_prompt": "<long detailed English prompt>", "wh_ratio": "16:9"}

preceded by a `<think>` block. Feeding the rewritten prompt to the image model is the
officially documented two-stage workflow; this module is the local implementation of the
first stage.

Why it runs on CPU (and why fp8-on-GPU is a dead end here)
==========================================================
16.68 GiB of BF16 text weights cannot live in the 15.82 GiB of VRAM on this box, and this
model must never share VRAM with the fp8 image transformer either (6.63 GiB resident). So the
rewriter is staged *before* the image model exactly like the Qwen3-VL text encoder is: load,
rewrite, free, and only then touch the GPU.

The obvious idea - reuse `fp8_quant.py`, which is what makes the image transformer fit - was
tried and measured, and it is a trap for *this* model:

  * the fp8 model is 10.24 GiB and does fit (10.31 GiB resident), but that leaves ~2.1 GiB of
    free VRAM, and the 2426-token system prompt prefill needs more: `dmesg` filled with
    `dxgkio_create_allocation: Ioctl failed: -12` (ENOMEM) and the prefill never finished;
  * more importantly it is not even faster. A forward pass costs 4.7 s at 32 tokens, 3.0 s at
    128, 3.6 s at 512, 4.4 s at 1024 and 6.2 s at 2048 - i.e. flat in sequence length, so the
    time is the per-call fp8 -> bf16 dequantisation of 248 layers, not the GEMMs. (~12.4 GiB
    of weight conversion per token; `cache_dequant` would fix it but needs another 12.4 GiB.)

The image transformer gets away with weight-only fp8 because it runs 20-40 forward passes over
minutes; a 9B language model runs one forward per token, so that fixed cost dominates. CPU bf16
is therefore both the safe and the faster option here: AVX512-BF16 on the Ryzen 5 9600X gives
~6 tok/s prefill and a memory-bandwidth-bound ~0.4 tok/s decode (~2.5 s/token).

A bf16 **hybrid** split (`--device cuda --quantize none`, accelerate `device_map="auto"`) was
tried too and is also a dead end on this card: transformers 5.17 keeps the CPU-offloaded half on
`meta` and streams it per forward, the split lands at 8.75 GiB GPU / 7.93 GiB CPU, and a
sequential stack runs on its slowest device - prefill was still going after 7 minutes. Full
record in NOTES.md section 37.

So `--device auto` (the default) resolves to **CPU** here, and `--device cuda` is kept only so
the experiments stay reproducible and so a GPU with >17 GiB (where bf16 fits outright) can use
it.

Memory: the checkpoint is bigger than this VM's free RAM headroom by design, so the load
path uses `low_cpu_mem_usage` (empty-init + copy one shard at a time) and the caller is
expected to free it before loading anything else.

Usage
=====
    $QIP_VENV/bin/python scripts/pe_rewrite.py --prompt "一只在雨中弹吉他的柯基"
    $QIP_VENV/bin/python scripts/pe_rewrite.py --prompt "..." --json-out rw.json
    $QIP_VENV/bin/python scripts/pe_rewrite.py --prompt "..." --print-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

GIB = 2**30

# Aspect ratio -> render size, exactly as published in the model card. These are the sizes
# the rewriter is trained to recommend (2048x2048 and up), which is 4x the pixel count of a
# 1024x1024 run. On this box the pixel budget is the binding constraint (see NOTES.md), so
# treat this table as advice, not as a resolution to feed straight into the pipeline.
SIZE_BY_RATIO = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "Qwen", "Qwen-Image-2.1-PE-T2I",
)

# Sampling defaults straight from the model card. They are part of the recipe: the model
# was tuned to emit its JSON under exactly this distribution, so do not "improve" them.
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20


def _avail_gib() -> float:
    """*Available* RAM in GiB - MemAvailable, not MemFree.

    `SC_AVPHYS_PAGES` is MemFree, which excludes reclaimable page cache. Right after the
    18.84 GB download, most of those pages are the model file itself: MemFree read 15.4 GiB
    while MemAvailable was 22.2 GiB, and gating an 18.8 GiB load on the former refuses a load
    that fits comfortably. This is the NOTES.md sections 28/30 mistake in a new place - the
    number has to mean what the check assumes it means.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB -> GiB
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB


def load_rewriter(model_dir: str = DEFAULT_MODEL, dtype: torch.dtype = torch.bfloat16,
                  threads: int | None = None, require_gib: float = 0.0,
                  device: str = "auto", quantize: str = "fp8"):
    """Load the rewriter. Returns (model, tokenizer, system_prompt).

    ``require_gib`` is a caller-supplied floor of *available* RAM (16.68 GiB of text weights
    in bf16; ignoring the 333 vision tensors saves 2.2 GiB), checked before anything is
    allocated so a doomed load fails with an explanation instead of the OOM killer.

    ``device='auto'`` prefers the GPU *with fp8 weights*. Plain bf16 cannot fit (16.68 GiB
    into ~13 GiB free), but as fp8 storage - exactly the `fp8_quant.py` scheme the image
    transformer already uses - the linear layers halve to ~7 GB and it does. If the fp8 model
    does not fit either, it falls back to CPU rather than gambling on an OOM: an out-of-memory
    on this box has taken the Windows host down before.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"rewriter not found: {model_dir}\n"
            f"fetch it with:  bash scripts/download_pe.sh"
        )
    if threads:
        torch.set_num_threads(threads)

    avail = _avail_gib()
    print(f"[pe] loading {model_dir}\n[pe] available RAM {avail:.2f} GiB", flush=True)
    if require_gib and avail < require_gib:
        raise SystemExit(
            f"[pe] refusing to load: need ~{require_gib:.1f} GiB available RAM, "
            f"have {avail:.2f} GiB. Free the image-model weights first."
        )

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dev = device
    if dev == "auto":
        # CPU on purpose: measured on this box, fp8-on-GPU is ~1-4 s per *forward* (roughly
        # flat in sequence length, so it is layer overhead, not compute) and cannot hold the
        # 2426-token system-prompt prefill. bf16 CPU is both safer and more predictable here.
        # See the module docstring.
        dev = "cpu"

    if dev == "cuda" and quantize == "none":
        # bf16 hybrid: 16.68 GiB does not fit in ~13.6 GiB of VRAM, so `accelerate` keeps as
        # much as fits on the GPU and spills the tail of the stack to CPU. The spill lands on
        # lm_head (the checkpoint's largest single tensor, 1.02B params) which is also what we
        # want: its prefill logits are [seq, 248320] bf16 = 1.2 GiB at 2451 tokens, and that
        # allocation is what killed the fp8 prefill earlier. On CPU it costs ~20 ms/token.
        #
        # MEASURED (NOTES.md section 37): this does not pay off on a 15.82 GiB card. The split
        # lands at ~8.75 GiB GPU / 7.93 GiB CPU, transformers 5.17 leaves the CPU half on `meta`
        # and streams it per forward, and a sequential stack runs at its slowest device's pace -
        # prefill was still running after 7 minutes. Kept for reproducibility on a bigger GPU.
        print("[pe] WARNING: bf16 hybrid did not pay off on this card - see NOTES.md section 37; "
              "the CPU default is faster here.", flush=True)
        free = torch.cuda.mem_get_info()[0] / GIB
        gpu_cap = max(2, int(free - 2.0))          # leave 2 GiB for activations
        cpu_cap = max(2, int(_avail_gib() - 2))
        max_memory = {0: f"{gpu_cap}GiB", "cpu": f"{cpu_cap}GiB"}
        print(f"[pe] bf16 hybrid: gpu_cap={gpu_cap} GiB, cpu_cap={cpu_cap} GiB", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map="auto", max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        model.eval()
        placed = {}
        for name, p in model.named_parameters():
            key = str(p.device)
            placed[key] = placed.get(key, 0) + p.numel() * p.element_size()
        print("[pe] device split: " + ", ".join(f"{k} {v/GIB:.2f} GiB" for k, v in placed.items()),
              flush=True)
        params = sum(p.numel() for p in model.parameters())
        print(f"[pe] {params/1e9:.2f}B params, loaded in {time.perf_counter()-t0:.1f}s "
              f"| device hybrid(gpu+cpu)", flush=True)
        return model, tokenizer, _system_prompt(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"[pe] {params/1e9:.2f}B params, {params*2/GIB:.2f} GiB, "
          f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    if dev == "cuda":
        if quantize == "fp8":
            from fp8_quant import fp8_memory_summary, quantize_linears_
            t1 = time.perf_counter()
            # lm_head is skipped on purpose: at 1.017B params its dequantised bf16 buffer
            # alone is 1.9 GiB, and `dequantize_transposed` goes through fp32 first (3.8 GiB),
            # which is most of the remaining VRAM. It stays bf16, like the embeddings.
            report = quantize_linears_(
                model, min_params=1 << 16, skip_names=("lm_head",))
            summary = fp8_memory_summary(model)
            print(f"[pe] quantised {report['converted']} linears "
                  f"({report['skipped']} skipped) in {time.perf_counter()-t1:.1f}s; "
                  f"model now {summary['total_GiB']:.2f} GiB", flush=True)
        else:
            summary = {"total_GiB": params * 2 / GIB, "fp8_layers": 0}
        free, total = torch.cuda.mem_get_info()
        need = summary["total_GiB"] + 1.5
        if free / GIB < need:
            print(f"[pe] GPU needs ~{need:.2f} GiB but only {free/GIB:.2f} GiB "
                  f"of {total/GIB:.2f} GiB is free -> CPU", flush=True)
            dev = "cpu"
        else:
            t1 = time.perf_counter()
            model = model.to(dev)
            torch.cuda.synchronize()
            print(f"[pe] moved to {dev} in {time.perf_counter()-t1:.1f}s; "
                  f"{(free - torch.cuda.mem_get_info()[0])/GIB:.2f} GiB resident", flush=True)

    sys_path = os.path.join(model_dir, "system_prompt.txt")
    if not os.path.isfile(sys_path):
        raise SystemExit(f"[pe] system_prompt.txt missing from {model_dir} - incomplete download?")
    with open(sys_path, encoding="utf-8") as fh:
        system_prompt = fh.read().strip()
    print(f"[pe] system prompt {len(system_prompt)} chars | device {dev}", flush=True)
    return model, tokenizer, system_prompt


def _system_prompt(model_dir: str) -> str:
    path = os.path.join(model_dir, "system_prompt.txt")
    if not os.path.isfile(path):
        raise SystemExit(f"[pe] system_prompt.txt missing from {model_dir} - incomplete download?")
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def build_inputs(tokenizer, system_prompt: str, user_prompt: str, thinking: bool = True):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )
    return tokenizer(text, return_tensors="pt")


def parse_output(text: str) -> dict:
    """Split `<think>...</think>` from the answer and decode the JSON object.

    The answer is supposed to be a bare JSON object, but a sampled 9B will occasionally
    wrap it in a markdown fence or add a sentence, so scan for the first balanced `{...}`
    instead of trusting `json.loads(text)`.
    """
    thinking, _, answer = text.partition("</think>")
    answer = answer.strip()
    if not answer:
        # No closing tag: the model ran out of budget mid-thought (or thinking is off).
        answer = text.strip()
        thinking = ""

    candidate = answer
    fence = re.search(r"```(?:json)?\s*(.*?)```", answer, re.S)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {answer[:400]!r}")
    obj = json.loads(candidate[start:end + 1])
    if "rewritten_prompt" not in obj:
        raise ValueError(f"JSON object has no rewritten_prompt: {obj!r}")
    obj.setdefault("wh_ratio", "1:1")
    out = {"rewritten_prompt": obj["rewritten_prompt"], "wh_ratio": obj["wh_ratio"]}
    if thinking:
        out["thinking"] = thinking.strip()
    return out


def rewrite(model, tokenizer, system_prompt: str, user_prompt: str, *,
            max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE,
            top_p: float = DEFAULT_TOP_P, top_k: int = DEFAULT_TOP_K,
            seed: int = 42, thinking: bool = True) -> dict:
    """Run one rewrite. Returns the parsed dict plus timing/token stats."""
    inputs = build_inputs(tokenizer, system_prompt, user_prompt, thinking=thinking)
    dev = torch.device(getattr(model, "device", "cpu"))
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]
    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kw.update(temperature=temperature, top_p=top_p, top_k=top_k)
    if seed is not None:
        torch.manual_seed(seed)

    print(f"[pe] prompt {len(user_prompt)} chars -> {n_in} tokens; generating "
          f"(max {max_new_tokens}) ...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    dt = time.perf_counter() - t0
    new_ids = out[0, n_in:]
    n_out = int(new_ids.shape[0])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"[pe] {n_out} tokens in {dt:.1f}s ({n_out/dt:.2f} tok/s)", flush=True)

    result = parse_output(text)
    result["stats"] = {
        "input_tokens": n_in, "output_tokens": n_out,
        "seconds": round(dt, 1), "tok_per_s": round(n_out / dt, 2) if dt else None,
        "truncated": n_out >= max_new_tokens,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("QIP_PE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--prompt", required=True, help="short request, any language")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="the card allows 16256; on CPU each token is ~0.3-0.5 s, so the "
                         "default is a budget, not the card's ceiling")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto: GPU with fp8 weights when it fits, else CPU")
    ap.add_argument("--quantize", default="fp8", choices=["fp8", "none"],
                    help="fp8 is what makes the GPU fit at all (halves the linear weights)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="skip the <think> block (faster, lower quality)")
    ap.add_argument("--print-thinking", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="timing probe: generate --max-new-tokens and skip parsing")
    ap.add_argument("--json-out", default=None, help="write {rewritten_prompt, wh_ratio} here")
    ap.add_argument("--with-size", action="store_true",
                    help="also print the card's recommended width/height")
    args = ap.parse_args()

    model, tokenizer, system_prompt = load_rewriter(
        args.model, threads=args.threads or None, require_gib=20.0,
        device=args.device, quantize=args.quantize)
    if args.bench:
        # Timing probe: no parsing, so a truncated (max_new_tokens=32) run still reports
        # tok/s and load time without dying on a JSON error. Decides the real budget.
        inputs = build_inputs(tokenizer, system_prompt, args.prompt, thinking=args.thinking)
        dev = torch.device(getattr(model, "device", "cpu"))
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        from transformers import LogitsProcessor

        class _Tick(LogitsProcessor):
            """Print a timestamp per step: separates prefill cost from decode rate.

            A single total number cannot tell a 4-minute prefill from 2 s/token decode, and
            the two have completely different fixes (the hybrid stack's reference
            delta-rule kernels make both plausible).
            """

            def __init__(self):
                self.t0 = time.perf_counter()
                self.n = 0

            def __call__(self, input_ids, scores):
                self.n += 1
                print(f"[pe]   step {self.n:3d}  +{time.perf_counter()-self.t0:6.1f}s",
                      flush=True)
                return scores

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, top_k=args.top_k,
                                 pad_token_id=tokenizer.pad_token_id,
                                 logits_processor=[_Tick()])
        dt = time.perf_counter() - t0
        n = int(out.shape[1] - inputs["input_ids"].shape[1])
        print(f"[pe] bench {n} tokens in {dt:.1f}s = {n/dt:.2f} tok/s "
              f"(prefill {inputs['input_ids'].shape[1]} tokens)", flush=True)
        return 0
    result = rewrite(
        model, tokenizer, system_prompt, args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, seed=args.seed, thinking=args.thinking,
    )

    thinking = result.pop("thinking", None)
    if args.print_thinking and thinking:
        print("\n" + "=" * 78 + "\nTHINKING\n" + "=" * 78)
        print(thinking)
    print("\n" + "=" * 78 + "\nRESULT\n" + "=" * 78)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.with_size:
        w, h = SIZE_BY_RATIO.get(result["wh_ratio"], (2048, 2048))
        print(f"\ncard's render size for {result['wh_ratio']}: {w}x{h} "
              f"(this box renders 1024x1024 - see NOTES.md)")
    if args.json_out:
        payload = {k: result[k] for k in ("rewritten_prompt", "wh_ratio")}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
