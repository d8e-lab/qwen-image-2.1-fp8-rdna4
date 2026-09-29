#!/usr/bin/env python3
"""Persistent Qwen-Image-2.1 server with a web UI, for iterating on prompts.

Loads the fp8 transformer and VAE once and keeps them resident on the GPU, so
generating another image costs only the denoise loop (~140 s for 1024x1024/40 steps)
instead of also paying 27 s of transformer load plus quantisation on every run.

The text encoder is NOT kept resident. At 16.33 GiB it cannot coexist with the
resident transformer in ~14 GiB of VRAM, so it is loaded, used for prompt encoding,
and freed per request. That is affordable because it loads from page cache in ~2 s
against minutes of denoising.

Usage:
    source scripts/env.sh
    $QIP_VENV/bin/python scripts/server.py                 # 127.0.0.1:7860
    $QIP_VENV/bin/python scripts/server.py --port 7861 --share

The UI exposes prompt, size, steps, guidance, seed and negative prompt. Generations
are serialised with a lock: two concurrent denoise loops would not fit in VRAM.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import (  # noqa: E402
    GIB,
    TextEncoderPool,
    build_pipeline_shell,
    encode_prompt,
    load_transformer,
    load_vae,
    log,
)


class Generator:
    """Holds the resident models and turns a prompt into an image."""

    def __init__(self, model_dir: str, dtype: torch.dtype, quantize: bool, device: str,
                 wait_for_vram_s: float = 120.0):
        self.model_dir = model_dir
        self.dtype = dtype
        self.device = device
        self.lock = threading.Lock()

        log("building pipeline shell (configs only) ...")
        self.pipe = build_pipeline_shell(model_dir, dtype)

        log("loading + quantising transformer ...")
        tf = load_transformer(model_dir, dtype, quantize=quantize)
        self.pipe.transformer = tf

        if device == "cuda":
            from fp8_quant import fp8_memory_summary
            need = fp8_memory_summary(tf)["total_GiB"]

            # A long-lived server should not die because something else briefly held the
            # GPU (a browser, an editor, another job). Wait for the memory instead, and
            # only give up after a sustained shortage.
            deadline = time.time() + wait_for_vram_s
            while True:
                free = torch.cuda.mem_get_info()[0] / GIB
                if free >= need + 0.5:
                    break
                if time.time() >= deadline:
                    raise SystemExit(
                        f"not enough VRAM after waiting {wait_for_vram_s:.0f}s: "
                        f"need ~{need + 0.5:.2f} GiB, have {free:.2f} GiB. Close GPU-using "
                        f"apps on Windows (browsers, games, other WSL sessions) and retry."
                    )
                log(f"waiting for VRAM: {free:.2f} GiB free, need {need + 0.5:.2f} GiB ...")
                time.sleep(5)

            log(f"moving transformer to {device} (free {free:.2f} GiB, need {need:.2f} GiB)")
            self.pipe.transformer.to(device)
            torch.cuda.synchronize()
            log(f"transformer resident, {torch.cuda.mem_get_info()[0]/GIB:.2f} GiB free")

        log("loading vae ...")
        self.pipe.vae = load_vae(model_dir, dtype)
        self.pipe.vae.to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        if device == "cuda":
            log(f"vae resident, {torch.cuda.mem_get_info()[0]/GIB:.2f} GiB free")

        # Tiling bounds the VAE decode, which is the single largest allocation in a run.
        self.pipe.vae.enable_tiling()
        self.encoder = TextEncoderPool(model_dir, dtype)
        log("ready")

    def generate(self, prompt: str, negative_prompt: str, height: int, width: int,
                 steps: int, guidance: float, seed: int):
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")
        steps = int(steps)

        with self.lock:                       # one denoise loop at a time
            t_all = time.perf_counter()

            log(f"encoding prompt ({len(prompt)} chars) ...")
            t0 = time.perf_counter()
            with self.encoder as te:
                emb, mask, _ = encode_prompt(self.pipe, te, prompt, device="cpu")
            log(f"encoded in {time.perf_counter()-t0:.1f}s -> {tuple(emb.shape)}")
            emb = emb.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            neg_emb = neg_mask = None
            if negative_prompt and negative_prompt.strip():
                t0 = time.perf_counter()
                with self.encoder as te:
                    neg_emb, neg_mask, _ = encode_prompt(
                        self.pipe, te, negative_prompt, device="cpu")
                log(f"negative encoded in {time.perf_counter()-t0:.1f}s")
                neg_emb = neg_emb.to(self.device)
                if neg_mask is not None:
                    neg_mask = neg_mask.to(self.device)
            # Free the encoder's host RAM before the denoise loop allocates.
            import gc
            gc.collect()

            seq = (int(width) // 16) * (int(height) // 16)
            cache_gib = 32 * 2 * seq * 4096 * 2 / GIB * (2 if guidance > 1 else 1)
            if self.device == "cuda":
                free = torch.cuda.mem_get_info()[0] / GIB
                use_kv = cache_gib < free - 2.2
                log(f"kv cache needs {cache_gib:.2f} GiB, free {free:.2f} GiB -> "
                    f"{'on' if use_kv else 'off'}")
            else:
                use_kv = False

            # Tiling only matters for large images; keep it on when it is cheap.
            if max(int(height), int(width)) < 768:
                self.pipe.vae.disable_tiling()
            else:
                self.pipe.vae.enable_tiling()

            gen = torch.Generator(self.device).manual_seed(int(seed))
            log(f"denoising {width}x{height}, {steps} steps, guidance={guidance} ...")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.pipe(
                    prompt=None,
                    prompt_embeds=emb,
                    prompt_embeds_mask=mask,
                    negative_prompt_embeds=neg_emb,
                    negative_prompt_embeds_mask=neg_mask,
                    true_cfg_scale=float(guidance),
                    height=int(height),
                    width=int(width),
                    num_inference_steps=steps,
                    generator=gen,
                    use_kv_cache=use_kv,
                    output_type="pil",
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            log(f"denoised in {dt:.1f}s ({dt/steps:.2f}s/step)")

            img = out.images[0]
            log(f"total {time.perf_counter()-t_all:.1f}s")
            del emb, mask, neg_emb, neg_mask
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return img


def build_ui(gen: Generator, default_size: int, default_steps: int):
    import gradio as gr

    def run(prompt, negative, size, steps, guidance, seed):
        try:
            img = gen.generate(prompt, negative, size, size, steps, guidance, seed)
            return img, f"OK - {size}x{size}, {steps} steps"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"ERROR: {type(e).__name__}: {e}"

    with gr.Blocks(title="Qwen-Image-2.1 (fp8)") as demo:
        gr.Markdown(
            "# Qwen-Image-2.1 · fp8 · RX 9070 XT\n"
            "模型常驻显存，改提示词后直接重新生成即可，无需重新加载。"
        )
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="提示词 Prompt", lines=3,
                                    value="a red apple on a wooden table")
                negative = gr.Textbox(label="负向提示词 Negative prompt", lines=2,
                                      value="")
                with gr.Row():
                    size = gr.Slider(256, 1024, value=default_size, step=64,
                                     label="边长 Size (正方形)")
                    steps = gr.Slider(2, 60, value=default_steps, step=1, label="步数 Steps")
                with gr.Row():
                    guidance = gr.Slider(1.0, 4.0, value=1.0, step=0.1,
                                         label="Guidance (true_cfg_scale；>1 更慢)")
                    seed = gr.Number(value=42, label="随机种子 Seed", precision=0)
                btn = gr.Button("生成 Generate", variant="primary")
                status = gr.Textbox(label="状态", interactive=False)
            with gr.Column(scale=3):
                gallery = gr.Image(label="结果", type="pil", height=640)
        btn.click(run, [prompt, negative, size, steps, guidance, seed], [gallery, status])
        prompt.submit(run, [prompt, negative, size, steps, guidance, seed], [gallery, status])
        gr.Markdown(
            "参考耗时：512×512/15 步约 57 s；1024×1024/40 步约 138 s（均为实测，不含加载）。"
        )
    return demo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--quantize", default="fp8", choices=["fp8", "none"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--size", type=int, default=512, help="default image side")
    ap.add_argument("--steps", type=int, default=15, help="default step count")
    ap.add_argument("--wait-for-vram", type=float, default=120.0,
                    help="seconds to wait for free VRAM at startup before giving up")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    log("=" * 70)
    log("Qwen-Image-2.1 persistent server")
    log("=" * 70)
    log(f"torch {torch.__version__} | device {args.device} | quantize {args.quantize}")

    gen = Generator(args.model, dtype, args.quantize == "fp8", args.device,
                    wait_for_vram_s=args.wait_for_vram)
    demo = build_ui(gen, args.size, args.steps)
    log(f"serving on http://{args.host}:{args.port}")
    demo.queue().launch(server_name=args.host, server_port=args.port,
                        share=args.share, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
