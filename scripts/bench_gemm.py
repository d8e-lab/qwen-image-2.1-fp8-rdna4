"""Find out how fast this stack can actually do a matmul, and on which backend.

The earlier benchmark showed 64x1024x4096 bf16 taking >1 s, i.e. <1 GFLOP/s, which is
not a real GPU kernel time. That points at a fallback path (rocBLAS Tensile / unoptimized
gemm) being selected instead of a tuned kernel. This script separates:
  - raw square matmuls across sizes (mm)
  - the actual linear shapes used by the Qwen-Image transformer (addmm)
and reports achieved GFLOP/s so the bottleneck is unambiguous.
"""

import os
import time

import torch
import torch.nn.functional as F

torch.cuda.init()
torch.manual_seed(0)
dev = "cuda"

print("TORCH_BLAS_PREFER_HIPBLASLT =", os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "<unset>"))
print("torch", torch.__version__)
props = torch.cuda.get_device_properties(0)
print("gpu", props.name, "| CUs", props.multi_processor_count)


def bench(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def report(label, n, k, m, dt, kind="mm"):
    a = torch.randn(n, k, device=dev, dtype=dt).mul(0.05)
    b = torch.randn(k, m, device=dev, dtype=dt).mul(0.05)
    if kind == "mm":
        fn = lambda: a @ b
    else:
        fn = lambda: F.linear(a, b.t())
    t = bench(fn)
    flops = 2.0 * n * k * m
    print(
        f"{label:28s} {kind:5s} {n}x{k}x{m:<7d} {str(dt).split('.')[-1]:9s} "
        f"{t*1000:9.3f} ms  {flops/t/1e9:9.2f} GFLOP/s",
        flush=True,
    )
    del a, b
    torch.cuda.empty_cache()


print("\n--- square mm (fp32) ---")
for s in [512, 1024, 2048]:
    report("square", s, s, s, torch.float32)

print("\n--- square mm (bf16) ---")
for s in [512, 1024, 2048]:
    report("square", s, s, s, torch.bfloat16)

print("\n--- square mm (fp16) ---")
for s in [512, 1024, 2048]:
    report("square", s, s, s, torch.float16)

print("\n--- transformer-like linear (bf16) ---")
for (tokens, cin, cout) in [
    (64, 1024, 4096),
    (512, 4096, 12288),
    (4096, 4096, 12288),
    (16384, 4096, 12288),
]:
    report("linear", tokens, cin, cout, torch.bfloat16, kind="linear")

print("\n--- transformer-like linear (fp32) ---")
report("linear", 4096, 4096, 12288, torch.float32, kind="linear")
