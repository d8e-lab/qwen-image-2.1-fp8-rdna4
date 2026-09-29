#!/usr/bin/env python3
"""Sustained-load probe: is the GPU throttling, or is throughput uniformly lower?

Memory bandwidth and raw matmul throughput diverge in a meaningful way. If bandwidth is
normal but matmul is slow, the compute units are clocked down or contended, rather than the
whole GPU being starved.
"""
import time
import torch

torch.cuda.init()
LOG = open("/home/helck/workspace/qwen-image/logs/throttle.log", "w")

def p(*a):
    print(*a, file=LOG, flush=True)

a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
fl = 2 * 4096 ** 3

p("sustained 4096^3 bf16 matmul, 1 s buckets:")
torch.cuda.synchronize()
for sec in range(10):
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        a @ b
        n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p("  bucket %2d: %4d iters  %7.2f TFLOP/s" % (sec + 1, n, n * fl / dt / 1e12))

x = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
y = torch.empty_like(x)
for _ in range(3):
    y.copy_(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(12):
    y.copy_(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 12
p("bandwidth: %7.1f GB/s" % (2 * (1 << 28) / dt / 1e9))
p("DONE")
