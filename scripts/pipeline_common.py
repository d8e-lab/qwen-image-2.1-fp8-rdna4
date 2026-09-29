#!/usr/bin/env python3
"""Shared model loading for Qwen-Image-2.1, used by both the CLI and the server.

Why this module exists: `run_qwen_image.py` and a persistent server need the same
non-obvious loading rules, and duplicating them is how the two drift apart. The rules,
each learned from a failure, are:

* **Load the pipeline from configs, not with `DiffusionPipeline.from_pretrained`.**
  That helper loads every component named in `model_index.json` at once, which is a
  ~21 GiB host-RAM peak (transformer 13.25 GiB bf16 + text encoder 16.33 GiB) against
  ~21 GiB available. It also cannot express "load only some components": a missing
  `model_index.json` entry trips its expected-modules check, and a `None` entry crashes
  its own loader.
* **fp8 weights must be quantised on CPU, then moved once.**
* **The text encoder cannot coexist with the resident transformer** (16.33 GiB vs
  ~8 GiB free VRAM), so it is loaded, used for encoding, and freed on every request.
* **A transposed-view linear weight is ~1000x slower** than a contiguous `(in, out)`
  one. `Fp8Weight` handles this; never call `F.linear` on these weights.
"""

from __future__ import annotations

import gc
import json
import os
import time

import torch

from fp8_quant import fp8_memory_summary, quantize_linears_

GIB = 2**30


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pipeline_shell(model_dir: str, dtype: torch.dtype):
    """Construct an empty QwenImage21Pipeline (processor + scheduler only, no weights)."""
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    with open(os.path.join(model_dir, "model_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    sched_dir = os.path.join(model_dir, "scheduler")
    if not os.path.isfile(os.path.join(sched_dir, "scheduler_config.json")):
        raise SystemExit(f"scheduler/scheduler_config.json missing under {model_dir}")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)

    proc_dir = os.path.join(model_dir, "processor")
    if not os.path.isfile(os.path.join(proc_dir, "tokenizer.json")):
        raise SystemExit(f"processor/tokenizer.json missing under {model_dir}")
    processor = Qwen3VLProcessor.from_pretrained(proc_dir)

    return QwenImage21Pipeline(
        scheduler=scheduler, vae=None, text_encoder=None,
        processor=processor, transformer=None,
    )


def load_transformer(model_dir: str, dtype: torch.dtype, quantize: bool = True,
                     cache_dequant: bool = False, min_params: int = 1 << 16):
    """Load the transformer on CPU and quantise it (weights stay on CPU)."""
    from diffusers import QwenImage21Transformer2DModel

    t0 = time.perf_counter()
    tf = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(model_dir, "transformer"), torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    log(f"transformer loaded on CPU in {time.perf_counter()-t0:.1f}s")

    if quantize:
        t0 = time.perf_counter()
        rep = quantize_linears_(tf, cache_dequant=cache_dequant, min_params=min_params)
        gc.collect()
        sm = fp8_memory_summary(tf)
        log(f"fp8: {rep['converted']} layers -> {sm['total_GiB']:.2f} GiB "
            f"in {time.perf_counter()-t0:.1f}s")
        if sm["fp8_layers"] == 0:
            raise SystemExit("fp8 quantisation produced no layers")
    return tf


def load_vae(model_dir: str, dtype: torch.dtype):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(
        os.path.join(model_dir, "vae"), torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    return vae


class TextEncoderPool:
    """Loads the text encoder on demand, encodes, and frees it.

    The encoder is 16.33 GiB in bf16 and only needed before denoising, so it is never
    kept resident. Loading it from page cache is fast (measured ~2 s), which is why a
    persistent server can afford to do this per request rather than holding it.
    """

    def __init__(self, model_dir: str, dtype: torch.dtype):
        self.model_dir = model_dir
        self.dtype = dtype
        self._mod = None

    def __enter__(self):
        from transformers import Qwen3VLForConditionalGeneration

        t0 = time.perf_counter()
        self._mod = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(self.model_dir, "text_encoder"),
            torch_dtype=self.dtype, low_cpu_mem_usage=True,
        ).eval()
        log(f"text encoder loaded ({time.perf_counter()-t0:.1f}s)")
        return self._mod

    def __exit__(self, *exc):
        self._mod = None
        gc.collect()
        return False


def encode_prompt(pipe, text_encoder, prompt: str, device: str = "cpu",
                  image=None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the pipeline's own encode_prompt.

    Keep the pipeline object around for this: it derives the prompt template and the
    number of leading system tokens from the real processor, and it documents that
    `prompt_embeds_mask` is None when the mask is entirely valid (the normal case for a
    single text prompt) and that `image_pad_mask` is an all-False tensor for text-to-image.
    """
    pipe.text_encoder = text_encoder
    try:
        with torch.no_grad():
            emb, mask, img_mask = pipe.encode_prompt(
                prompt=prompt, image=image, device=device, num_images_per_prompt=1
            )
    finally:
        pipe.text_encoder = None
    return emb, mask, img_mask
