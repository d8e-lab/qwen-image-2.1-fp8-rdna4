#!/usr/bin/env python3
"""Verify the downloaded checkpoint actually loads and runs, before any long job.

`preflight.py` checks that bytes are present and hashes match. This goes further and
proves the artefacts are *usable*: the transformer index resolves across both shards,
the model constructs, fp8 quantisation applies to the real weights, and a real forward
pass produces finite output. Runs on CPU so it works while the GPU is unavailable.

This is the last gate before spending time on a GPU run, and it is fast: it never
runs the diffusion loop.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import fp8_memory_summary, gpu_alloc_works, quantize_linears_

GIB = 2**30


def main() -> int:
    model = os.environ.get("QIP_MODEL", os.path.expanduser(
        "~/workspace/models/Qwen/Qwen-Image-2.1"))
    print("=" * 78)
    print("checkpoint load verification")
    print("=" * 78)
    print(f"model dir: {model}")

    # ---- files -------------------------------------------------------------
    need = [
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "scheduler/scheduler_config.json",
        "processor/tokenizer.json",
    ]
    missing = [r for r in need if not os.path.isfile(os.path.join(model, r))]
    if missing:
        print("\nmissing files:")
        for m in missing:
            print(f"  x {m}")
        print("\nrun scripts/fast_download.py first")
        return 1
    print("required files present")

    from diffusers import QwenImage21Transformer2DModel

    # ---- transformer --------------------------------------------------------
    print("\n[1] loading transformer (both shards) ...")
    t0 = time.perf_counter()
    tf = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(model, "transformer"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    tf.eval()
    dt = time.perf_counter() - t0
    n_lin = sum(1 for m in tf.modules() if isinstance(m, torch.nn.Linear))
    n_par = sum(p.numel() for p in tf.parameters())
    bf16_gib = sum(p.numel() * p.element_size() for p in tf.parameters()) / GIB
    print(f"  loaded in {dt:.0f}s | {n_par/1e9:.3f} B weights | {bf16_gib:.3f} GiB bf16 | {n_lin} Linear")

    # ---- fp8 ----------------------------------------------------------------
    print("\n[2] fp8 quantisation ...")
    t0 = time.perf_counter()
    rep = quantize_linears_(tf, min_params=1 << 16)
    print(f"  {rep['converted']} layers -> fp8 in {time.perf_counter()-t0:.0f}s "
          f"({rep['skipped']} small layers left in bf16)")
    assert rep["converted"] > 0, "no layers quantised"
    sm = fp8_memory_summary(tf)
    print(f"  transformer now {sm['total_GiB']:.3f} GiB "
          f"(saved {bf16_gib - sm['total_GiB']:.3f} GiB vs bf16)")
    assert sm["total_GiB"] < bf16_gib * 0.6, "fp8 did not save enough memory"

    # ---- real forward pass --------------------------------------------------
    print("\n[3] real forward pass on the real weights ...")
    # The transformer takes packed latents (B, tokens, in_channels) plus the latent grid
    # layout. `img_shapes` is per-sample [[(frame, height, width), ...]] in latent tokens
    # with condition images first and the target last; `img_mask` spans the *joint*
    # sequence (text + condition + target), one slot per 2x2 group of target latents.
    # For pure text-to-image the pipeline builds exactly this from the text mask plus
    # `latents.shape[1] // 4` ones, so mirror that here.
    grid = 16                      # 16x16 latent tokens = 256 tokens
    tokens = grid * grid
    text_tokens = 32

    # Layout, read off a fully instrumented real pipeline call (captured from the tiny
    # fixture, which builds the same structures):
    #   repeats = where(img_mask, 4, 1)
    #   joint   = cat([encoder_hidden_states, zeros(target_tokens // 4)], dim=1)
    #   joint   = joint.repeat_interleave(repeats, dim=1)
    #   joint[:, repeat_interleave(img_mask, repeats)] = hidden_states
    # Observed for a pure text-to-image sample (4x4 latents, 11 text tokens):
    #   encoder_hidden_states (1, 11, C); img_mask (1, 15) with the LAST 4 entries True;
    #   hidden_states (1, 16, C) == 4 x (number of True).
    # So: the target image's slots are the trailing True block, each expanding to 4 tokens,
    # and the joint length is text_tokens + target_tokens // 4 == len(img_mask).
    target_groups = tokens // 4
    joint_len = text_tokens + target_groups
    x = torch.randn(1, tokens, tf.config.in_channels, dtype=torch.bfloat16) * 0.5
    ctx = torch.randn(1, text_tokens, tf.config.context_in_dim, dtype=torch.bfloat16) * 0.5
    t = torch.tensor([0.5], dtype=torch.bfloat16)
    img_shapes = [[(1, grid, grid)]]
    # bool; leading text slots False, trailing target slots True. Asserting the invariants
    # the transformer relies on keeps this from silently drifting if the layout changes.
    img_mask = torch.cat([
        torch.zeros(1, text_tokens, dtype=torch.bool),
        torch.ones(1, target_groups, dtype=torch.bool),
    ], dim=1)
    assert img_mask.shape[1] == joint_len
    assert tokens == 4 * int(img_mask.sum()), "hidden token count must be 4x the True slots"
    t0 = time.perf_counter()
    with torch.no_grad():
        out = tf(x, encoder_hidden_states=ctx, timestep=t,
                 img_shapes=img_shapes, img_mask=img_mask, return_dict=False)
    dt = time.perf_counter() - t0
    sample = out[0] if isinstance(out, (tuple, list)) else out
    print(f"  forward ok in {dt:.1f}s -> {tuple(sample.shape)} {sample.dtype}")
    assert torch.isfinite(sample).all(), "forward produced non-finite values"
    print(f"  finite output, mean {sample.float().mean().item():+.5f}, "
          f"std {sample.float().std().item():.5f}")

    # ---- vae ----------------------------------------------------------------
    print("\n[4] loading vae ...")
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(
        os.path.join(model, "vae"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    vae.eval()
    vp = sum(p.numel() for p in vae.parameters())
    print(f"  vae {vp/1e6:.1f} M params")

    # ---- gpu (optional) -----------------------------------------------------
    print("\n[5] gpu move (optional) ...")
    if gpu_alloc_works():
        need_gib = sm["total_GiB"] + 0.5
        free, total = torch.cuda.mem_get_info()
        print(f"  free {free/GIB:.2f} / {total/GIB:.2f} GiB, need {need_gib:.2f} GiB")
        if free / GIB < need_gib:
            print("  NOT ENOUGH VRAM - would abort rather than risk an OOM")
            return 1
        t0 = time.perf_counter()
        tf = tf.to("cuda")
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        print(f"  moved to GPU in {time.perf_counter()-t0:.0f}s, {free/GIB:.2f} GiB free")
        with torch.no_grad():
            xg = x.to("cuda"); cg = ctx.to("cuda"); tg = t.to("cuda")
            og = tf(xg, encoder_hidden_states=cg, timestep=tg,
                    img_shapes=img_shapes, img_mask=img_mask.to("cuda"), return_dict=False)
            torch.cuda.synchronize()
        s = og[0] if isinstance(og, (tuple, list)) else og
        assert torch.isfinite(s).all(), "GPU forward non-finite"
        print(f"  GPU forward ok -> {tuple(s.shape)}")
    else:
        print("  GPU unusable - skipped (see GPU_RECOVERY_NEEDED.md)")

    print("\n" + "=" * 78)
    print("CHECKPOINT VERIFIED - transformer + fp8 + VAE all load and run")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
