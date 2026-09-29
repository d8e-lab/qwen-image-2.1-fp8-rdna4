"""Validate the fp8 quantiser on a real Qwen-Image-2.1 transformer config, CPU only.

Building the actual ``QwenImage21Transformer2DModel`` from its config exercises the
real layer layout (32 single-stream blocks, ModuleList ``to_out``, SwiGLU MLP,
modulation MLP) instead of the toy module in test_fp8.py, and gives the true
memory numbers the VRAM budget is based on.

Runs entirely on CPU so it works even while the GPU is unavailable. Before the
first matmul it *checks* whether CUDA can allocate and skips the device tests if
not, because a failed allocation poisons the caching allocator on this stack.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import Fp8Weight, fp8_memory_summary, gpu_alloc_works, quantize_linears_

GIB = 2**30


def main() -> int:
    from diffusers import QwenImage21Transformer2DModel

    cfg = {
        "_class_name": "QwenImage21Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "context_in_dim": 4096,
        "in_channels": 64,
        "num_attention_heads": 32,
        "num_layers": 32,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 3,
        "eps": 1e-06,
        "causal_condition": True,
    }
    print("building real-shape QwenImage21Transformer2DModel on CPU ...")
    t0 = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_config(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    orig_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / GIB
    print(f"  built in {time.perf_counter()-t0:.1f}s")
    print(f"  parameters : {n_params/1e9:.3f} B  (README claims 7B for the visual component)")
    print(f"  fp32 bytes : {orig_gib:.3f} GiB")

    # Cast to bf16 first, like the pipeline does, then quantise.
    model = model.to(torch.bfloat16)
    bf16_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / GIB
    print(f"  bf16 bytes : {bf16_gib:.3f} GiB   <- what must fit in 15.8 GiB VRAM unquantised")

    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"  nn.Linear  : {n_linear}")

    t0 = time.perf_counter()
    report = quantize_linears_(model, cache_dequant=False)
    dt = time.perf_counter() - t0
    print(f"\nquantised {report['converted']} layers ({report['skipped']} skipped) in {dt:.1f}s")
    print(f"  linear weights: {report['orig_bytes']/GIB:.3f} GiB -> {report['fp8_bytes']/GIB:.3f} GiB "
          f"({report['orig_bytes']/max(report['fp8_bytes'],1):.2f}x)")

    sm = fp8_memory_summary(model)
    print(f"\ntransformer total after fp8: {sm['total_GiB']:.3f} GiB "
          f"({sm['fp8_layers']} fp8 layers, {sm['other_params']} other tensors)")
    print(f"  vs bf16                   {bf16_gib:.3f} GiB  "
          f"(saved {bf16_gib - sm['total_GiB']:.3f} GiB)")

    assert sm["total_GiB"] < bf16_gib * 0.65, "fp8 did not save enough"
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules() if not isinstance(m, Fp8Weight))

    # Numerical sanity on a single real block-sized linear.
    lin = next(m for m in model.modules() if isinstance(m, Fp8Weight))
    print(f"\nsample layer: {lin}")
    x = torch.randn(64, lin.in_features, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        y = lin(x)
    print(f"  forward ok: {tuple(x.shape)} -> {tuple(y.shape)} dtype={y.dtype}")
    assert y.shape == (64, lin.out_features)
    assert torch.isfinite(y).all(), "non-finite output"

    # ---- CUDA checks, only if the driver is healthy -----------------------
    print("\nchecking whether CUDA can allocate ...")
    if gpu_alloc_works():
        print("  CUDA OK -> testing device move + real forward")
        model_small = model
        before = next(m for m in model_small.modules() if isinstance(m, Fp8Weight)).qweight.clone()
        model_small = model_small.to("cuda")
        after = next(m for m in model_small.modules() if isinstance(m, Fp8Weight))
        assert after.qweight.dtype == torch.uint8, after.qweight.dtype
        assert after.qweight.is_cuda
        assert torch.equal(after.qweight.cpu(), before), "bytes changed in transit"
        with torch.no_grad():
            xc = x.to("cuda")
            yc = after(xc)
            torch.cuda.synchronize()
        print(f"  cuda forward ok: {tuple(yc.shape)} {yc.dtype}")
        free, total = torch.cuda.mem_get_info()
        print(f"  vram after move: free {free/GIB:.2f} / {total/GIB:.2f} GiB")
    else:
        print("  CUDA NOT USABLE (host GPU memory pool is exhausted) - skipping device tests")
        print("  this is an environment state, not a code defect; a WSL restart clears it")

    print("\nREAL-SHAPE CPU VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
