"""Measure fp8 weight-only storage vs bf16 on this GPU.

Three candidate strategies for the Qwen-Image-2.1 transformer (7.1B params):
  A. bf16 baseline                                  -> 14.27 GB weights
  B. fp8 weights, dequant to bf16 every forward     ->  7.13 GB weights, extra dequant per call
  C. fp8 weights + LRU cache of dequantized bf16    ->  7.13 GB floor, cache trades VRAM for speed

hipBLASLt is broken on this WSL2/librocDXG stack, so shapes are kept small and the
env var TORCH_BLAS_PREFER_HIPBLASLT=0 is set by env.sh to skip the failed-lookup retry.
"""
import json
import time

import torch
import torch.nn.functional as F

torch.cuda.init()
torch.manual_seed(0)
dev = "cuda"

OUT_JSON = "/home/helck/workspace/qwen-image/logs/bench_fp8.json"


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    res = {}
    for (Tn, C, OUT) in [(64, 1024, 4096), (128, 2048, 8192), (256, 4096, 12288)]:
        x = torch.randn(Tn, C, device=dev, dtype=torch.bfloat16).mul(0.05)
        W = torch.randn(OUT, C, device=dev, dtype=torch.bfloat16).mul(0.02)
        W8 = W.to(torch.float8_e4m3fn)

        tb = bench(lambda: F.linear(x, W))
        tf = bench(lambda: F.linear(x, W8.to(torch.bfloat16)))

        # accuracy of fp8 weight-only, per-tensor vs per-row scaling
        ref = F.linear(x.float(), W.float())
        deq_t = W8.to(torch.bfloat16)
        row_scale = (W.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp(min=1e-12)
        W8r = (W.float() / row_scale).to(torch.float8_e4m3fn)
        deq_r = (W8r.float() * row_scale).to(torch.bfloat16)
        err_t = (F.linear(x, deq_t).float() - ref).abs().max().item() / ref.abs().max().item()
        err_r = (F.linear(x, deq_r).float() - ref).abs().max().item() / ref.abs().max().item()

        key = f"{Tn}x{C}x{OUT}"
        res[key] = {
            "bf16_ms": tb,
            "fp8deq_ms": tf,
            "overhead_x": tf / tb,
            "w_bf16_MiB": W.numel() * 2 / 2**20,
            "w_fp8_MiB": W.numel() / 2**20,
            "rel_err_pertensor": err_t,
            "rel_err_perrow": err_r,
        }
        print(
            f"{key:20s} bf16={tb:7.3f}ms fp8+deq={tf:7.3f}ms x{tf/tb:4.2f}  "
            f"W {W.numel()*2/2**20:.0f}->{W.numel()/2**20:.0f} MiB  "
            f"err perT={err_t:.2e} perRow={err_r:.2e}",
            flush=True,
        )
        del x, W, W8, W8r
        torch.cuda.empty_cache()

    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print("done ->", OUT_JSON)


if __name__ == "__main__":
    main()
