#!/usr/bin/env python3
"""Build a tiny on-disk Qwen-Image-2.1 look-alike so the *real* runner can be tested.

`run_qwen_image.py` has a lot of disk-layout-dependent logic (subfolder names, the
scheduler's `scheduler_config.json` vs components' `config.json`, staged component
loading, fp8 round-trip, KV-cache sizing). Testing it against the real 33 GB
checkpoint is slow and, while the GPU is down, impossible. This fixture writes a
complete but miniature model tree with the same layout, using the real tokenizer so
the processor works, and the real class names so `from_pretrained` hits the same
code paths.

Not a substitute for the real run — it validates wiring, not image quality.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import torch

OUT = os.path.expanduser("~/workspace/qwen-image/tiny_model")
REAL = os.environ.get("QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1"))

DIM = 64
CTX = 64
HEADS = 4
HEAD_DIM = 16
LAYERS = 3


def main() -> int:
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Transformer2DModel,
    )
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    os.makedirs(OUT, exist_ok=True)
    print(f"writing tiny model tree -> {OUT}")

    # ---- transformer --------------------------------------------------------
    tf = QwenImage21Transformer2DModel.from_config({
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
    }).to(torch.bfloat16).eval()
    tf.save_pretrained(os.path.join(OUT, "transformer"), safe_serialization=True)
    print(f"  transformer  {sum(p.numel() for p in tf.parameters())/1e6:.2f} M params")

    # ---- vae ----------------------------------------------------------------
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
    }).to(torch.float32).eval()
    vae.save_pretrained(os.path.join(OUT, "vae"), safe_serialization=True)
    print(f"  vae          {sum(p.numel() for p in vae.parameters())/1e6:.2f} M params")

    # ---- text encoder -------------------------------------------------------
    te = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
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
    )).to(torch.bfloat16).eval()
    te.save_pretrained(os.path.join(OUT, "text_encoder"))
    print(f"  text_encoder {sum(p.numel() for p in te.parameters())/1e6:.2f} M params")

    # ---- scheduler ----------------------------------------------------------
    FlowMatchEulerDiscreteScheduler().save_pretrained(os.path.join(OUT, "scheduler"))
    print("  scheduler    saved")

    # ---- processor: reuse the real tokenizer files ---------------------------
    pdir = os.path.join(OUT, "processor")
    os.makedirs(pdir, exist_ok=True)
    src = os.path.join(REAL, "processor")
    copied = []
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
              "preprocessor_config.json", "video_preprocessor_config.json"):
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(pdir, f))
            copied.append(f)
    print(f"  processor    {len(copied)} files copied from the real tokenizer")
    if "tokenizer.json" not in copied:
        raise SystemExit(f"real tokenizer not found under {src}")

    # ---- model_index.json ---------------------------------------------------
    idx = {
        "_class_name": "QwenImage21Pipeline",
        "_diffusers_version": "0.41.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    with open(os.path.join(OUT, "model_index.json"), "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(OUT) for f in fs
    )
    print(f"\ntiny model tree complete: {total/2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
