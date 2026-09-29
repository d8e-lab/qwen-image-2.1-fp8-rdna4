#!/usr/bin/env python3
"""Validate the GPU path the moment the driver is healthy again.

The one thing that has never once succeeded on this machine is moving fp8 weights to
the GPU, so this settles it in a throwaway process before any large model is loaded.
Order matters: the cheapest decisive test first, expensive model loads last, so a
failure is reported in seconds rather than after a 7 GiB load.

Every probe runs in a subprocess. A failed HIP allocation leaves the parent's caching
allocator asserting (`!handles_.at(i) INTERNAL ASSERT FAILED`), so an in-process probe
would poison everything that follows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

GIB = 2**30
RESULTS: list[tuple[str, bool, str]] = []


def run_probe(name: str, body: str, timeout: int = 300) -> bool:
    """Run `body` in a fresh interpreter; record and print the outcome."""
    code = "import torch\n" + textwrap.dedent(body)
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
    )
    ok = r.returncode == 0
    detail = ""
    if ok:
        # last non-empty stdout line is the probe's own summary
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        detail = lines[-1][:120] if lines else "ok"
    else:
        err = [l for l in r.stderr.strip().splitlines() if l.strip()]
        detail = " | ".join(err[-2:])[:200] if err else f"exit {r.returncode}"
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}", flush=True)
    return ok


def main() -> int:
    print("=" * 78)
    print("GPU validation for Qwen-Image-2.1 fp8")
    print("=" * 78)
    import torch
    print(f"torch {torch.__version__} | hip {torch.version.hip}")
    print(f"TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT', '<unset>')}")
    print()

    print("[1] can the GPU allocate at all?")
    if not run_probe("tiny bf16 alloc", """
        torch.cuda.init()
        t = torch.zeros(8, dtype=torch.bfloat16, device='cuda')
        torch.cuda.synchronize()
        print('ok')
    """):
        print("\nGPU is still unusable. Most likely cause: PYTORCH_HIP_ALLOC_CONF")
        print("contains expandable_segments, which breaks allocation on this ROCm build.")
        print("If it is unset, reset the AMD adapter on Windows (Restart-GPU.ps1).")
        return 1

    run_probe("device properties", """
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f'{p.name} | free {free/2**30:.2f}/{total/2**30:.2f} GiB')
    """)

    print("\n[2] the decisive fp8 question")
    run_probe("raw fp8 dtype -> cuda", """
        t = torch.randn(64, 64).to(torch.float8_e4m3fn)
        g = t.to('cuda')
        torch.cuda.synchronize()
        print('raw float8_e4m3fn moved ok')
    """)
    run_probe("uint8 bytes -> cuda -> view(fp8)", """
        src = (torch.randn(256, 128) * 0.05)
        q = src.to(torch.float8_e4m3fn)
        u = q.view(torch.uint8)
        g = u.to('cuda')
        back = g.view(torch.float8_e4m3fn)
        # Compare against a CPU-side dequant: comparing the GPU result to `src` directly
        # raises "Expected all tensors to be on the same device" and was a bug in this
        # probe, not in Fp8Weight.
        deq_cpu = q.float()
        deq_gpu = back.float().cpu()
        err = (deq_gpu - deq_cpu).abs().max().item()
        assert err == 0.0, f'bytes changed in transit: {err}'
        print(f'uint8->cuda->view(fp8) bit-exact (max err {err})')
    """)
    run_probe("Fp8Weight module -> cuda forward", """
        import sys, os
        sys.path.insert(0, os.path.expanduser('~/workspace/qwen-image/scripts'))
        import torch.nn as nn
        from fp8_quant import quantize_linears_
        m = nn.Sequential(nn.Linear(128, 256, bias=False), nn.Linear(256, 64, bias=False))
        quantize_linears_(m, min_params=0)
        m = m.to('cuda')
        x = torch.randn(4, 128, dtype=torch.bfloat16, device='cuda')
        y = m(x)
        torch.cuda.synchronize()
        print(f'fp8 module on GPU ok, out {tuple(y.shape)} {y.dtype}')
    """)
    run_probe("fp8 dtype survives .to(bf16)", """
        import sys, os
        sys.path.insert(0, os.path.expanduser('~/workspace/qwen-image/scripts'))
        import torch.nn as nn
        from fp8_quant import Fp8Weight, quantize_linears_
        m = nn.Sequential(nn.Linear(64, 64, bias=False))
        quantize_linears_(m, min_params=0)
        before = m[0].qweight.clone()
        m = m.to(torch.bfloat16)
        assert m[0].qweight.dtype == torch.uint8, m[0].qweight.dtype
        assert m[0].scale.dtype == torch.float32, m[0].scale.dtype
        assert torch.equal(m[0].qweight, before)
        print('fp8 bytes + fp32 scale preserved')
    """)

    print("\n[3] compute backends")
    run_probe("hipBLASLt-left-off matmul speed", """
        import time, torch.nn.functional as F
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda') * 0.05
        b = torch.randn(12288, 4096, dtype=torch.bfloat16, device='cuda') * 0.05
        # NOTE: F.linear with an (out, in) weight takes a degenerate path on this stack;
        # 0.1 TFLOP/s vs 84 for a contiguous (in, out) weight. Fp8Weight avoids it by
        # materialising the weight transposed and using `x @ Wt`, which is what this
        # probe measures so the number reflects the real model.
        bt = b.t().contiguous()
        for _ in range(2): a @ bt
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): a @ bt
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        tflops = 2 * 4096 * 4096 * 12288 / (ms / 1000) / 1e12
        print(f'x @ Wt: {ms:.2f} ms, {tflops:.1f} TFLOP/s (expect ~80-130)')
    """)
    run_probe("fp8 dequant-on-the-fly overhead", """
        import time, torch.nn.functional as F
        x = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda') * 0.05
        w = torch.randn(12288, 4096, dtype=torch.bfloat16, device='cuda') * 0.02
        w8 = w.to(torch.float8_e4m3fn)
        def bench(fn, n=5):
            for _ in range(2): fn()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000
        tb = bench(lambda: F.linear(x, w))
        tf = bench(lambda: F.linear(x, w8.to(torch.bfloat16)))
        print(f'bf16 {tb:.2f} ms vs fp8+dequant {tf:.2f} ms (x{tf/tb:.2f})')
    """)
    run_probe("scaled_dot_product_attention", """
        import torch.nn.functional as F
        q = torch.randn(1, 32, 4096, 128, dtype=torch.bfloat16, device='cuda') * 0.1
        o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        print(f'sdpa ok {tuple(o.shape)}')
    """)

    print("\n[4] fp8 quantised transformer at real shape (7.1B)")
    model_dir = os.environ.get(
        "QIP_MODEL", os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
    )
    tf_dir = os.path.join(model_dir, "transformer")
    if os.path.isdir(tf_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(tf_dir)
    ):
        run_probe("load + quantise + move real transformer", f"""
            import sys, os, time, gc
            sys.path.insert(0, os.path.expanduser('~/workspace/qwen-image/scripts'))
            from fp8_quant import quantize_linears_, fp8_memory_summary
            from diffusers import QwenImage21Transformer2DModel
            t0 = time.perf_counter()
            m = QwenImage21Transformer2DModel.from_pretrained({tf_dir!r}, torch_dtype=torch.bfloat16)
            print(f'  loaded bf16 in {{time.perf_counter()-t0:.0f}}s')
            rep = quantize_linears_(m)
            gc.collect()
            s = fp8_memory_summary(m)
            print(f'  quantised {{rep["converted"]}} layers, {{s["total_GiB"]:.2f}} GiB')
            free, _ = torch.cuda.mem_get_info()
            need = s['total_GiB'] * 2**30 + 2**30
            assert free > need, f'not enough VRAM: {{free/2**30:.2f}} < {{need/2**30:.2f}} GiB'
            t0 = time.perf_counter()
            m = m.to('cuda')
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info()
            print(f'  moved to GPU in {{time.perf_counter()-t0:.0f}}s, {{free/2**30:.2f}} GiB free')
            del m
        """, timeout=1800)
    else:
        print("  SKIP  real transformer not downloaded yet")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(RESULTS)} probes passed")
    blocked = [n for n, ok, _ in RESULTS if not ok]
    if blocked:
        print("failing: " + ", ".join(blocked))
        print("=" * 78)
        return 1
    print("GPU path fully validated.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
