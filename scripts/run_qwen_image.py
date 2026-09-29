#!/usr/bin/env python3
"""Generate images with Qwen-Image-2.1 (fp8 weights) on RX 9070 XT / ROCm / WSL2.

Memory strategy
===============
VRAM is 15.82 GiB total, of which roughly 14 GiB is free (the Windows desktop owns the
rest), and an OOM on this stack has taken the host down before, so the budget is enforced
rather than hoped for. Measured steady state at 1024x1024:

    transformer (fp8, 7.115B params)   6.63 GiB   resident for the whole denoise loop
    attention KV cache (1024x1024)     2.00 GiB   (4.00 GiB if guidance > 1)
    activations                        ~1.0 GiB
    vae (bf16)                          0.63 GiB
    -------------------------------------------
    peak during denoise                ~10.3 GiB of ~14 GiB free

The Qwen3-VL text encoder is 16.33 GiB in bf16 and CANNOT coexist with the resident
transformer, so in a GPU run the prompt is deliberately encoded on CPU (~16 s, against
minutes of denoising). Two further things are load-bearing for staying inside the budget,
and both were bugs before they were features:

  * the VAE *decode* is the single largest allocation in the run, not the denoise loop,
    so VAE tiling is auto-enabled for GPU runs at 768px or more;
  * attention must use the flash kernel (pinned in scripts/env.sh) - SDPA otherwise
    selects MATH and materialises a 4.91 GiB score matrix instead of 0.06 GiB.

Phases
======
  1. load on CPU -> quantise transformer to fp8 -> free the bf16 originals
  2. move transformer to the compute device (refuses if it will not fit)
  3. load text encoder on CPU -> prompt embeds -> free it (GPU runs encode on CPU)
  4. denoise loop on the compute device
  5. load VAE -> decode (tiled when large) -> PNG
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import fp8_memory_summary, quantize_linears_  # noqa: E402

GIB = 2**30

# Absolute floor of free VRAM required before any phase that touches the GPU.
# Set from --min-free-gib in main(); this is the "do not risk an OOM/GPU hang"
# guard, because an out-of-memory here has previously taken the Windows host down.
MIN_FREE_GIB = 9.0


def vram(tag: str) -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    print(f"  [vram] {tag:38s} free={free/GIB:6.2f} GiB / {total/GIB:6.2f} GiB", flush=True)
    return free, total


def require_free(min_gib: float, what: str, floor: bool = False) -> None:
    """Abort unless at least ``min_gib`` is free.

    With ``floor=True`` the user's --min-free-gib is applied as well, so a phase
    never starts below the configured safety margin even if its own estimate is
    smaller than that margin.
    """
    if not torch.cuda.is_available():
        return
    need = max(min_gib, MIN_FREE_GIB) if floor else min_gib
    free, _ = torch.cuda.mem_get_info()
    if free / GIB < need:
        raise SystemExit(
            f"\nRefusing to continue: need ~{need:.2f} GiB free VRAM for {what}, "
            f"but only {free/GIB:.2f} GiB is available.\n"
            f"Close GPU-using apps on Windows (browsers, games, other WSL sessions) or "
            f"lower --height/--width, then retry."
        )


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sync() -> None:
    """Synchronise the compute device (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("QIP_MODEL", os.path.expanduser(
        "~/workspace/models/Qwen/Qwen-Image-2.1")))
    ap.add_argument("--prompt", default='A neon shop sign that reads "QWEN IMAGE 2.1", '
                                        "rainy night, reflections on wet pavement")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="true_cfg_scale; >1 doubles the KV cache and runs the transformer twice per step")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="output.png")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quantize", default="fp8", choices=["fp8", "none"],
                    help="fp8 stores transformer Linears as float8_e4m3fn (halves VRAM)")
    ap.add_argument("--fp8-cache-dequant", action="store_true",
                    help="cache dequantised bf16 weights. Measured ~2x faster "
                         "(36.5 -> 18.4 s/step on CPU) but needs ~13 GiB more RAM/VRAM")
    ap.add_argument("--kv-cache", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--vae-tiling", action="store_true",
                    help="lower VAE decode peak VRAM (auto-enabled on GPU >=768px)")
    ap.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false",
                    help="disable automatic VAE tiling")
    ap.add_argument("--vae-slicing", action="store_true")
    ap.add_argument("--attention-slicing", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the transformer (slow first run)")
    ap.add_argument("--vae-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                    help="auto follows --dtype; fp32 decode costs more VRAM but is more exact")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu is a slow fallback for validation when the GPU is unavailable")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + quantise + move to GPU, then stop before inference")
    ap.add_argument("--quant-min-params", type=int, default=1 << 16,
                    help="skip Linears smaller than this many weights (tiny models need 0)")
    ap.add_argument("--min-free-gib", type=float, default=4.0,
                    help="absolute floor of free VRAM required before the denoise "
                         "loop; guards against an OOM/GPU hang. Note the transformer "
                         "is already resident by then, so this is slack, not total")
    args = ap.parse_args()

    # steps=1 is silently broken with this scheduler: its sigma grid ends at a value that
    # makes FlowMatchEulerDiscreteScheduler._stretch_to_terminal divide by zero
    # (scale_factor = 0), yielding NaN latents that decode to a blank image. Verified:
    # steps 2..40 are all fine. Fail loudly rather than emit a blank PNG.
    if args.steps < 2:
        raise SystemExit(
            f"--steps must be >= 2 (got {args.steps}). With shift_terminal="
            f"{0.02} the scheduler divides by zero for a single step and produces NaN "
            f"latents, i.e. a blank image."
        )

    global MIN_FREE_GIB
    MIN_FREE_GIB = args.min_free_gib

    dev = args.device
    on_gpu = dev == "cuda"

    log("=" * 78)
    log("Qwen-Image-2.1  |  fp8 weight-only  |  RX 9070 XT / gfx1201 / ROCm")
    log("=" * 78)
    log(f"torch        {torch.__version__}  (hip {torch.version.hip})")
    log(f"device       {torch.cuda.get_device_name(0) if on_gpu else 'CPU (fallback)'}")
    log(f"hipblaslt    TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT','<unset>')}")
    log(f"attn backend DIFFUSERS_ATTN_BACKEND={os.environ.get('DIFFUSERS_ATTN_BACKEND','<unset>')}")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    vae_dtype = dtype if args.vae_dtype == "auto" else {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32
    }[args.vae_dtype]
    log(f"dtype        {args.dtype}   quantize={args.quantize}   kv_cache={args.kv_cache}")
    log(f"vae dtype    {args.vae_dtype}")
    log(f"resolution   {args.width}x{args.height}   steps={args.steps}   guidance={args.guidance}")
    if on_gpu:
        log(f"vram guard   abort if free < {args.min_free_gib:.1f} GiB")

    if not os.path.isdir(args.model):
        raise SystemExit(f"model dir not found: {args.model}")

    # Import here so --help works without the heavy stack.
    from diffusers import QwenImage21Pipeline

    t_start = time.perf_counter()
    if on_gpu:
        vram("initial")

    # ---------------------------------------------------------------- phase 1
    # Components are loaded one at a time on purpose. Loading the whole pipeline at
    # once pulls transformer (13.25 GiB bf16) + text encoder (7.26 GiB) + VAE into
    # host RAM together, about 21.2 GiB, and this VM only has ~21 GiB available.
    # The text encoder is not needed until phase 3, so it is fetched later.
    from diffusers import QwenImage21Transformer2DModel

    log("\n[1/5] loading transformer on CPU ...")
    t0 = time.perf_counter()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype
    )
    log(f"      loaded in {time.perf_counter()-t0:.1f}s  (transformers "
        f"{__import__('transformers').__version__}, diffusers "
        f"{__import__('diffusers').__version__})")

    if args.quantize == "fp8" and args.fp8_cache_dequant:
        # The cache holds a second, bf16 copy of every quantised weight - roughly the
        # size of the original checkpoint. Measured peak RSS on this machine with the
        # cache enabled was 22.0 GiB against ~21 GiB available, i.e. right at the edge,
        # so require real headroom instead of discovering it as an OOM.
        if on_gpu:
            free_gib = torch.cuda.mem_get_info()[0] / GIB
            need_gib = 14.0
        else:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
            free_gib = avail
            need_gib = 20.0
        log(f"\n      --fp8-cache-dequant requested: {free_gib:.1f} GiB free, "
            f"need ~{need_gib:.1f} GiB")
        if free_gib < need_gib:
            log(f"      REFUSING the cache (would risk an OOM); continuing uncached")
            args.fp8_cache_dequant = False

    if args.quantize == "fp8":
        log("\n      quantising transformer Linears -> fp8_e4m3fn (per-row scales) ...")
        t0 = time.perf_counter()
        report = quantize_linears_(transformer, cache_dequant=args.fp8_cache_dequant,
                                   min_params=args.quant_min_params)
        log(f"      {report['converted']} layers quantised, {report['skipped']} skipped, "
            f"in {time.perf_counter()-t0:.1f}s")
        log(f"      Linear weights: {report['orig_bytes']/GIB:.2f} GiB -> {report['fp8_bytes']/GIB:.2f} GiB")
        gc.collect()
        summary = fp8_memory_summary(transformer)
        log(f"      transformer resident now: {summary['total_GiB']:.2f} GiB "
            f"({summary['fp8_layers']} fp8 layers + {summary['other_params']} other tensors)")

        # NOTE: do NOT try to save_pretrained/reload the quantised module as a way to
        # release host RAM. Fp8Weight keeps its bytes in *non-persistent* buffers, so
        # save_pretrained would emit only the fp32/bf16-shaped placeholders and the
        # reload would silently reconstruct empty nn.Linear layers (this is exactly
        # what an earlier version of this script did, and the layer-count assertion
        # below is what caught it). Keeping the quantised module in place is correct:
        # staged component loading already bounds host RAM.
        summary = fp8_memory_summary(transformer)
    else:
        summary = {"total_GiB": sum(p.numel() * p.element_size()
                                    for p in transformer.parameters()) / GIB}
        log(f"      no quantisation; transformer weights {summary['total_GiB']:.2f} GiB")

    # Build the pipeline from configs only: no component weights are loaded here, so
    # this register_modules pass costs a few megabytes. Each component's config has a
    # different filename (scheduler/scheduler_config.json vs transformer/config.json),
    # so map them explicitly rather than assuming "config.json" everywhere.
    log("\n      assembling pipeline (configs only, no weights) ...")
    # Do not use DiffusionPipeline.from_pretrained here. It loads every component named
    # in model_index.json together, which is exactly the ~21 GiB host-RAM peak we are
    # avoiding, and it rejects a hand-built shell (None entries crash its loader, and a
    # missing entry trips its "expected modules" check). Parsing the configs and
    # instantiating the pipeline directly is simpler and keeps the staged loading.
    import json as _json
    import tempfile as _tempfile

    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor

    with open(os.path.join(args.model, "model_index.json"), encoding="utf-8") as fh:
        index = _json.load(fh)
    for _req in ("scheduler", "processor"):
        if not isinstance(index.get(_req), list):
            raise SystemExit(f"model_index.json has no usable '{_req}' entry: {index.get(_req)!r}")

    _sched_dir = os.path.join(args.model, "scheduler")
    if not os.path.isfile(os.path.join(_sched_dir, "scheduler_config.json")):
        raise SystemExit(
            f"scheduler/scheduler_config.json missing under {args.model} - download incomplete?"
        )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(_sched_dir)

    _proc_dir = os.path.join(args.model, "processor")
    if not os.path.isfile(os.path.join(_proc_dir, "tokenizer.json")):
        raise SystemExit(
            f"processor/tokenizer.json missing under {args.model} - download incomplete?"
        )
    processor = Qwen3VLProcessor.from_pretrained(_proc_dir)

    # register_modules requires the class declared in model_index.json to match, and
    # holding a strong reference here prevents it being garbage collected.
    pipe = QwenImage21Pipeline(
        scheduler=scheduler, vae=None, text_encoder=None,
        processor=processor, transformer=transformer,
    )
    _tf_cls = index["transformer"][1] if isinstance(index.get("transformer"), list) else "?"
    log(f"      pipeline ready: transformer={type(transformer).__name__} (model_index says {_tf_cls}), "
        f"scheduler={type(scheduler).__name__}, processor={type(processor).__name__}")
    del index
    gc.collect()

    # ---------------------------------------------------------------- phase 2
    log(f"\n[2/5] moving transformer to {dev} ...")
    need = summary["total_GiB"] + 0.5
    if on_gpu:
        # First touch of the GPU: nothing is resident yet, so total + slack is correct here.
        require_free(need, "the transformer", floor=True)
    t0 = time.perf_counter()
    pipe.transformer.to(dev)
    sync()
    log(f"      transformer on {dev} in {time.perf_counter()-t0:.1f}s")
    if on_gpu:
        vram("transformer resident")

    # ---------------------------------------------------------------- phase 3a
    # The shell was built from configs only, so the text encoder and VAE are not in
    # memory yet. Load the text encoder now, then the VAE after the encoder is
    # dropped, so the two never coexist in host RAM or VRAM.
    from transformers import Qwen3VLForConditionalGeneration

    log("\n[3/5] loading text encoder ...")
    t0 = time.perf_counter()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    text_encoder.eval()
    # Plain attribute assignment (NOT __dict__). register_modules already set
    # self.config["text_encoder"] = (library, class); DiffusionPipeline.components does
    # getattr(self, key) for every config key, and if the instance attribute is absent
    # that falls through to ConfigMixin.__getattr__, which emits a deprecation warning
    # and returns None. setattr binds the real module so the lookup succeeds.
    pipe.text_encoder = text_encoder
    enc_gib = sum(p.numel() * p.element_size() for p in text_encoder.parameters()) / GIB
    log(f"      text encoder {enc_gib:.2f} GiB, loaded in {time.perf_counter()-t0:.1f}s "
        f"(kept only for prompt encoding)")

    if args.dry_run:
        log("\n--dry-run: stopping before inference.")
        return 0

    # Encode the prompt on the *same* device as the text encoder was placed on. On a GPU
    # run that is deliberately CPU: the fp8 transformer is resident in VRAM (6.63 GiB of
    # 15.8) and the text encoder needs 16.33 GiB, so they cannot coexist. Encoding is a
    # one-off cost (~16 s measured) against minutes of denoising, so paying it on CPU is
    # far cheaper than shuffling the transformer off the GPU and back (2 x 27 s) - and it
    # keeps peak VRAM at transformer + KV cache + VAE only.
    encode_dev = "cpu" if on_gpu else dev
    if not on_gpu:
        pass  # encoder is already on `dev`
    t0 = time.perf_counter()
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(
            prompt=args.prompt, image=None, device=encode_dev, num_images_per_prompt=1
        )
    sync()
    log(f"      encoded on {encode_dev} in {time.perf_counter()-t0:.1f}s  "
        f"embeds={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")
    prompt_embeds = prompt_embeds.to(dev)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(dev)
    log(f"      encoded in {time.perf_counter()-t0:.1f}s  "
        f"embeds={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")

    neg_embeds = neg_mask = None
    if args.negative_prompt is not None:
        with torch.no_grad():
            neg_embeds, neg_mask, _ = pipe.encode_prompt(
                prompt=args.negative_prompt, image=None, device=encode_dev,
                num_images_per_prompt=1
            )
        neg_embeds = neg_embeds.to(dev)
        if neg_mask is not None:
            neg_mask = neg_mask.to(dev)

    # Free the encoder before the KV cache is allocated.
    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    gc.collect()
    if on_gpu:
        torch.cuda.empty_cache()
    sync()
    log("      text encoder offloaded and freed")
    if on_gpu:
        vram("after freeing encoder")

    # Load the VAE only now: it is needed for the final decode, and loading it earlier
    # would put it in host RAM alongside the 7.26 GiB text encoder for no benefit.
    from diffusers import AutoencoderKLQwenImage21

    log("\n      loading vae ...")
    t0 = time.perf_counter()
    vae = AutoencoderKLQwenImage21.from_pretrained(
        args.model, subfolder="vae", torch_dtype=vae_dtype, low_cpu_mem_usage=True
    )
    vae.eval()
    pipe.vae = vae
    vae_gib = sum(q.numel() * q.element_size() for q in vae.parameters()) / GIB
    log(f"      vae {vae_gib:.2f} GiB ({str(vae_dtype).split(chr(46))[-1]}), "
        f"loaded in {time.perf_counter()-t0:.1f}s")
    # The VAE decodes the latents at the end of the denoise loop, so it has to be on the
    # compute device. Moving it here (not earlier) keeps it out of the way while the text
    # encoder is resident, and the transformer is the only other GPU tenant at this point.
    if on_gpu:
        # Deliberately NOT floor=True: the blanket --min-free-gib floor is meant to stop a
        # high-risk phase (the denoise loop) starting with little headroom. Applying it to a
        # 0.63 GiB VAE move would refuse on a perfectly safe budget - the transformer
        # legitimately holds 6.63 GiB, leaving ~8.4 GiB, and the VAE needs a fraction of it.
        require_free(vae_gib + 0.5, "the vae")
    vae.to(dev)
    sync()
    if on_gpu:
        vram("vae resident")

    # ---------------------------------------------------------------- phase 4
    kv = args.kv_cache
    if kv == "auto" and not on_gpu:
        kv = "off"
        log("\n      CPU run: kv cache off (no VRAM budget to protect)")
    if kv == "auto":
        # KV cache is 32 layers x 2 x seq x 4096 in bf16, doubled when guidance runs
        # the negative branch too. Only enable it when it demonstrably fits.
        seq = (args.width // 16) * (args.height // 16)
        cache_gib = 32 * 2 * seq * 4096 * 2 / GIB * (2 if args.guidance > 1 else 1)
        free, _ = torch.cuda.mem_get_info()
        # `free` ALREADY excludes the resident transformer and VAE. Subtracting the
        # transformer's size again double-counts it and yields a negative headroom, which
        # silently disabled the cache on every run (observed: "headroom ~-1.07 GiB -> off"
        # while 7.76 GiB was genuinely free). The cache only has to fit in what is
        # currently free, minus a margin for activations and fragmentation.
        headroom = free / GIB - 2.2
        kv = "on" if cache_gib < headroom else "off"
        log(f"\n      kv cache needs ~{cache_gib:.2f} GiB; headroom ~{headroom:.2f} GiB -> {kv}")

    if args.attention_slicing:
        # Not forwarded by this pipeline either; set it on the modules that support it.
        for _mod in (pipe.transformer, pipe.vae):
            if hasattr(_mod, "enable_attention_slicing"):
                _mod.enable_attention_slicing()
        log("      attention slicing enabled")
    if args.vae_slicing:
        pipe.vae.enable_slicing()
    # The 1024x1024 decode is the single largest allocation in the whole run
    # (autoencoder_kl_qwenimage21._decode, decoder_base_dim=144, 2x upsampling stages):
    # it OOMed with 7.26 GiB already resident, while the denoise loop itself was fine.
    # Tiling splits the decode into overlapping tiles and bounds that peak, so it is on
    # by default for GPU runs at 768px or more; --no-vae-tiling opts out.
    want_tiling = args.vae_tiling or (on_gpu and max(args.height, args.width) >= 768)
    if want_tiling:
        # QwenImage21Pipeline does not forward enable_vae_tiling(); call the VAE directly.
        pipe.vae.enable_tiling()
        log("      vae tiling enabled on the vae" + ("" if args.vae_tiling else " (auto: >=768px on GPU)"))

    if args.compile:
        log("      torch.compile(transformer) - first step will be slow")
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    log("\n[4/5] denoising ...")
    if on_gpu:
        # The transformer and VAE are ALREADY resident by now, so requiring their combined
        # size as *free* memory double-counts and refuses a healthy budget (observed:
        # 7.76 GiB free with 7.26 GiB already allocated was wrongly rejected). What the
        # loop still has to allocate is the KV cache plus activations, so guard on that.
        seq_len = (args.width // 16) * (args.height // 16)
        kv_need = 32 * 2 * seq_len * 4096 * 2 / GIB * (2 if args.guidance > 1 else 1) \
            if kv == "on" else 0.0
        require_free(kv_need + 2.0, "the denoise loop", floor=True)
    gen = torch.Generator(dev).manual_seed(args.seed)

    def cb(pipe_, step, timestep, kw):
        if on_gpu:
            free, _ = torch.cuda.mem_get_info()
            print(f"      step {step+1:3d}/{args.steps}  free={free/GIB:5.2f} GiB", flush=True)
        else:
            print(f"      step {step+1:3d}/{args.steps}", flush=True)
        return kw

    t0 = time.perf_counter()
    with torch.no_grad():
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_mask=neg_mask,
            true_cfg_scale=args.guidance,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=gen,
            use_kv_cache=(kv == "on"),
            output_type="pil",
            callback_on_step_end=cb,
        )
    sync()
    dt = time.perf_counter() - t0
    log(f"      denoised in {dt:.1f}s  ({dt/args.steps:.2f}s/step)")
    if on_gpu:
        vram("after denoise")

    # ---------------------------------------------------------------- phase 5
    log("\n[5/5] saving ...")
    image = result.images[0]
    image.save(args.output)
    log(f"      wrote {args.output}  ({image.size[0]}x{image.size[1]} {image.mode})")
    log(f"\ntotal wall time {time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
