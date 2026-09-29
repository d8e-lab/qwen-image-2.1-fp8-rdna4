#!/usr/bin/env python3
"""End-to-end smoke test of the QwenImage21Pipeline inference path on a tiny model.

Purpose: de-risk the real run. The 33 GB checkpoint takes hours to fetch over this
link, so rather than discover a wiring bug at the end, this builds a miniature but
*structurally identical* pipeline (same classes, same scheduler, same real tokenizer)
and runs the whole path on CPU: prompt encoding -> fp8 quantisation -> denoising loop
-> VAE decode -> PIL image.

It exercises exactly the things that could be wrong in `run_qwen_image.py`:
  * `encode_prompt` argument/return contract
  * the transformer's block-causal prefill + KV cache path
  * `use_kv_cache=True/False` both working
  * the fp8 `Fp8Weight` modules running inside the real attention/MLP blocks
  * `_pack_latents` / `_unpack_latents` / scheduler stepping / VAE decode
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import Fp8Weight, fp8_memory_summary, quantize_linears_

# Small but shape-compatible with the real pipeline.
DIM = 64          # transformer inner dim
CTX = 64          # text-encoder hidden size -> must equal transformer context_in_dim
HEADS = 4
HEAD_DIM = 16
LAYERS = 3


def build_pipeline(model_dir: str):
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    tf_cfg = {
        "_class_name": "QwenImage21Transformer2DModel",
        "attention_head_dim": HEAD_DIM,
        "axes_dims_rope": [4, 6, 6],
        "context_in_dim": CTX,
        "in_channels": 64,
        "num_attention_heads": HEADS,
        "num_layers": LAYERS,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 2,
        "eps": 1e-06,
        "causal_condition": True,
    }
    transformer = QwenImage21Transformer2DModel.from_config(tf_cfg)

    vae = AutoencoderKLQwenImage21.from_config({
        "_class_name": "AutoencoderKLQwenImage21",
        "attn_scales": [],
        "base_dim": 8,
        "decoder_base_dim": 8,
        "dim_mult": [1, 2, 4],
        "dropout": 0.0,
        "in_channels": 4,
        "is_residual": False,
        "latents_mean": [0.0] * 64,
        "latents_std": [1.0] * 64,
        "num_res_blocks": 1,
        "out_channels": 4,
        "temperal_downsample": [False, True, True],
        "z_dim": 64,
    })

    text_cfg = Qwen3VLConfig(
        text_config=dict(
            hidden_size=CTX, intermediate_size=CTX * 2, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=151936, max_position_embeddings=4096,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4,
            in_channels=3, out_hidden_size=CTX, patch_size=16,
            spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=1024,
        ),
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
    )
    text_encoder = Qwen3VLForConditionalGeneration(text_cfg).eval()

    processor = Qwen3VLProcessor.from_pretrained(model_dir, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

    return QwenImage21Pipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder,
        processor=processor, transformer=transformer,
    )


def main() -> int:
    model_dir = os.environ.get("QIP_MODEL", os.path.expanduser(
        "~/workspace/models/Qwen/Qwen-Image-2.1"))
    if not os.path.isdir(os.path.join(model_dir, "processor")):
        raise SystemExit(f"processor/ not found under {model_dir} (download not far enough along?)")

    torch.manual_seed(0)
    print("building tiny pipeline ...")
    t0 = time.perf_counter()
    pipe = build_pipeline(model_dir)
    print(f"  built in {time.perf_counter()-t0:.1f}s")

    # --- prompt encoding path ------------------------------------------------
    print("\nencoding a prompt ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        emb, mask, img_mask = pipe.encode_prompt(
            prompt="a red cube on a white table", image=None, device="cpu", num_images_per_prompt=1
        )
    print(f"  encode_prompt OK in {time.perf_counter()-t0:.1f}s")
    print(f"  embeds {tuple(emb.shape)} {emb.dtype} | mask "
          f"{None if mask is None else tuple(mask.shape)} | "
          f"img_pad {None if img_mask is None else tuple(img_mask.shape)}")
    assert emb.shape[0] == 1 and emb.shape[-1] == CTX
    # encode_prompt nulls the attention mask when it is entirely valid. The image-pad
    # mask comes back as a tensor that is all-False for pure text-to-image (it only
    # marks real <|image_pad|> positions, which a text-only prompt has none of).
    assert mask is None or mask.shape == emb.shape[:2], mask
    assert img_mask is not None and img_mask.shape == emb.shape[:2], img_mask
    assert not img_mask.any(), "t2i prompt reported image tokens"
    assert torch.isfinite(emb).all(), "non-finite prompt embeds"

    # --- fp8 quantisation of the transformer ---------------------------------
    print("\nquantising transformer -> fp8 ...")
    report = quantize_linears_(pipe.transformer, min_params=0)
    print(f"  converted {report['converted']} layers ({report['skipped']} skipped)")
    assert report["converted"] > 0, "nothing was quantised"
    n_lin = sum(1 for m in pipe.transformer.modules() if isinstance(m, torch.nn.Linear))
    assert n_lin == 0, f"{n_lin} nn.Linear survived quantisation"

    # --- full denoise + decode loop, both KV settings ------------------------
    for kv in (True, False):
        print(f"\nrunning pipeline (height=64 width=64 steps=2 use_kv_cache={kv}) ...")
        gen = torch.Generator("cpu").manual_seed(42)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe(
                prompt="a red cube on a white table",
                height=64,
                width=64,
                num_inference_steps=2,
                generator=gen,
                use_kv_cache=kv,
                output_type="pil",
                true_cfg_scale=1.0,
            )
        dt = time.perf_counter() - t0
        img = out.images[0]
        print(f"  OK in {dt:.1f}s -> {img.size} {img.mode}")
        # The real VAE is 16x spatial; this miniature one downsamples by 4, so only
        # assert a sane, non-degenerate image rather than an exact size.
        assert img.mode == "RGB", img.mode
        assert img.size[0] > 0 and img.size[1] > 0, img.size
        assert len(set(img.getdata())) > 1, "image is a single flat colour"

    print("\nTINY PIPELINE END-TO-END TEST PASSED")
    print("(this validates the wiring; the real fp8 win is measured in test_fp8_real.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
