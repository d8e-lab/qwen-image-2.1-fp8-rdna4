#!/usr/bin/env python3
"""Qwen-Image-2.1 image editing (single or multiple reference images).

The same checkpoint does text-to-image and editing; editing is selected by passing
reference images. That happens in two places inside the pipeline:

  * the **text encoder** sees the image pixels, because Qwen3-VL 4B is a vision-language
    model - the reference becomes `<|vision_start|><|image_pad|><|vision_end|>` tokens
    interleaved with the prompt text, so the condition is understood jointly;
  * the **VAE** encodes the same resized image into latent tokens that are concatenated
    *before* the target latents in the transformer's sequence.

Key implementation detail for our memory budget: when `prompt_embeds` are supplied the
pipeline skips the text encoder entirely, but still consumes `image` for the VAE path.
That is exactly what lets us keep the encoder off the GPU - we encode the prompt (and the
reference, via the VLM) on CPU, free the encoder, then let the pipeline run the VAE and
the denoise loop.

Editing costs more memory than text-to-image: a 1024x1024 reference adds 4096 latent
tokens to the 4096 target tokens, so the sequence is twice as long and attention is
quadratic in it. VRAM guards are applied accordingly and the resolution is checked
before anything is loaded.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import gpu_alloc_works  # noqa: E402
from pipeline_common import (  # noqa: E402
    GIB,
    TextEncoderPool,
    build_pipeline_shell,
    encode_prompt,
    load_transformer,
    load_vae,
    log,
)

VAE_SCALE = 16


def edit(args) -> int:
    model_dir = os.path.expanduser(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    vae_dtype = dtype if args.vae_dtype == "auto" else {
        "bfloat16": torch.bfloat16, "float16": torch.float16,
        "float32": torch.float32}[args.vae_dtype]
    on_gpu = args.device == "cuda"

    if args.steps < 2:
        raise SystemExit("--steps must be >= 2 (1 step divides by zero in the scheduler)")

    # ---- inputs ------------------------------------------------------------
    paths = args.image if isinstance(args.image, list) else [args.image]
    if not paths:
        raise SystemExit("at least one --image is required for editing")
    refs = []
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"reference image not found: {p}")
        # The pipeline normalises to RGBA itself; match that here so the VLM and the
        # VAE see the same pixels (README: transparent refs keep their alpha).
        refs.append(PILImage.open(p).convert("RGBA"))
    w0, h0 = refs[-1].size
    log(f"{len(refs)} reference image(s): "
        + ", ".join(f"{os.path.basename(p)} {im.size}" for p, im in zip(paths, refs)))

    # Resolve the output size the way the pipeline does: keep the reference's aspect
    # ratio at roughly output_resolution**2 total area, then snap to /32.
    target_area = (args.output_resolution or max(args.height, args.width)) ** 2
    ratio = w0 / h0
    import math
    width = round(math.sqrt(target_area * ratio) / 32) * 32
    height = round(math.sqrt(target_area / ratio) / 32) * 32
    if args.height and args.width:
        height, width = args.height, args.width
    log(f"output size {width}x{height} (aspect from the last reference)")

    seq_target = (width // VAE_SCALE) * (height // VAE_SCALE)
    seq_ref = sum((im.size[0] // VAE_SCALE) * (im.size[1] // VAE_SCALE) for im in refs)
    log(f"latent tokens: {seq_ref} reference + {seq_target} target "
        f"(text-to-image would be {seq_target} alone)")

    if on_gpu and not gpu_alloc_works():
        raise SystemExit(
            "GPU cannot allocate. Check PYTORCH_HIP_ALLOC_CONF is unset "
            "(see GPU_RECOVERY_NEEDED.md)."
        )

    # ---- load --------------------------------------------------------------
    log("\n[1/4] building pipeline (configs only) ...")
    pipe = build_pipeline_shell(model_dir, dtype)

    log("[1/4] loading + quantising transformer ...")
    tf = load_transformer(model_dir, dtype, quantize=(args.quantize == "fp8"))
    pipe.transformer = tf

    if on_gpu:
        from fp8_quant import fp8_memory_summary
        need = fp8_memory_summary(tf)["total_GiB"] + 0.5
        free = torch.cuda.mem_get_info()[0] / GIB
        log(f"moving transformer to cuda (free {free:.2f}, need {need:.2f} GiB)")
        if free < need:
            raise SystemExit(f"not enough VRAM for the transformer: need {need:.2f} GiB")
        tf.to("cuda")
        torch.cuda.synchronize()
        log(f"transformer resident, {torch.cuda.mem_get_info()[0]/GIB:.2f} GiB free")

    # ---- encode the prompt + reference on CPU --------------------------------
    log("\n[2/4] encoding prompt + reference (VLM) on CPU ...")
    t0 = time.perf_counter()
    with TextEncoderPool(model_dir, dtype) as te:
        emb, mask, _ = encode_prompt(pipe, te, args.prompt, device="cpu", image=refs)
    log(f"encoded in {time.perf_counter()-t0:.1f}s -> {tuple(emb.shape)}")
    emb = emb.to(args.device) if on_gpu else emb
    if mask is not None:
        mask = mask.to(args.device) if on_gpu else mask

    neg_emb = neg_mask = None
    if args.negative_prompt:
        with TextEncoderPool(model_dir, dtype) as te:
            neg_emb, neg_mask, _ = encode_prompt(
                pipe, te, args.negative_prompt, device="cpu", image=refs)
        if on_gpu:
            neg_emb = neg_emb.to(args.device)
            if neg_mask is not None:
                neg_mask = neg_mask.to(args.device)
    gc.collect()

    # ---- vae ---------------------------------------------------------------
    log("\n[3/4] loading vae ...")
    pipe.vae = load_vae(model_dir, vae_dtype)
    if on_gpu:
        pipe.vae.to(args.device)
        torch.cuda.synchronize()
    # Editing's larger sequence leaves less room, so keep tiling on.
    pipe.vae.enable_tiling()

    # ---- denoise -----------------------------------------------------------
    total_tokens = seq_ref + seq_target
    cache_gib = 32 * 2 * total_tokens * 4096 * 2 / GIB * (2 if args.guidance > 1 else 1)
    if on_gpu:
        free = torch.cuda.mem_get_info()[0] / GIB
        use_kv = args.kv_cache != "off" and (
            args.kv_cache == "on" or cache_gib < free - 2.5)
        log(f"\n[4/4] kv cache needs {cache_gib:.2f} GiB, free {free:.2f} GiB -> "
            f"{'on' if use_kv else 'off'}")
    else:
        use_kv = False

    log(f"[4/4] denoising {width}x{height}, {args.steps} steps, guidance={args.guidance} ...")
    gen = torch.Generator(args.device).manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pipe(
            prompt=None,
            prompt_embeds=emb,
            prompt_embeds_mask=mask,
            negative_prompt_embeds=neg_emb,
            negative_prompt_embeds_mask=neg_mask,
            image=refs,                      # consumed by the VAE path
            true_cfg_scale=args.guidance,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            generator=gen,
            use_kv_cache=use_kv,
            output_type="pil",
        )
    if on_gpu:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    log(f"denoised in {dt:.1f}s ({dt/args.steps:.2f}s/step)")
    if on_gpu:
        log(f"vram free after denoise: {torch.cuda.mem_get_info()[0]/GIB:.2f} GiB")

    img = out.images[0]
    img.save(args.output)
    log(f"wrote {args.output} ({img.size[0]}x{img.size[1]} {img.mode})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Qwen-Image-2.1 image editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", action="append", required=True,
                    help="reference image path; repeat for multi-image editing")
    ap.add_argument("--prompt", required=True,
                    help='edit instruction, e.g. "Change the background to a sunset beach"')
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--output", default="edit_out.png")
    ap.add_argument("--model", default=os.environ.get(
        "QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")))
    ap.add_argument("--height", type=int, default=0, help="0 = derive from the reference")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--output-resolution", type=int, default=1024,
                    help="target area used to derive size when --height/--width are 0")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--vae-dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--quantize", default="fp8", choices=["fp8", "none"])
    ap.add_argument("--kv-cache", default="auto", choices=["auto", "on", "off"])
    args = ap.parse_args()
    return edit(args)


if __name__ == "__main__":
    raise SystemExit(main())
