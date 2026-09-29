"""Unit-test the fp8 weight-only quantiser without needing the 33 GB checkpoint.

Builds a miniature version of the Qwen-Image-2.1 block layout (including the
``nn.ModuleList([nn.Linear, nn.Dropout])`` shape used by ``to_out``) and checks that
quantisation preserves numerics, shrinks memory, and survives .to(device).
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_quant import (
    Fp8Weight,
    fp8_memory_summary,
    gpu_alloc_works,
    quantize_linears_,
    quantize_weight,
)


class MiniAttention(nn.Module):
    def __init__(self, dim=256, heads=4, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)])


class MiniFF(nn.Module):
    def __init__(self, dim=256, hidden=768):
        super().__init__()
        self.proj = nn.Linear(dim, hidden, bias=False)
        self.gate_layer = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)


class MiniBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attn = MiniAttention(dim)
        self.ff = MiniFF(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))


class MiniTransformer(nn.Module):
    def __init__(self, dim=256, depth=3):
        super().__init__()
        self.img_in = nn.Linear(64, dim, bias=False)
        self.blocks = nn.ModuleList([MiniBlock(dim) for _ in range(depth)])
        self.proj_out = nn.Linear(dim, 64, bias=False)

    def forward(self, x):
        h = self.img_in(x)
        for b in self.blocks:
            h = h + b.attn.to_out[0](b.attn.to_q(h)) * 0.0  # keep shapes simple
            h = h + b.ff.out(b.ff.proj(h))
            h = h + b.modulation(h)[..., : h.shape[-1]]
        return self.proj_out(h)


def main():
    torch.manual_seed(0)
    dim = 256
    m = MiniTransformer(dim=dim, depth=3).eval()

    n_linear = sum(1 for x in m.modules() if isinstance(x, nn.Linear))
    orig_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    print(f"built mini transformer: {n_linear} nn.Linear, {orig_bytes/2**20:.2f} MiB")
    assert n_linear == 26, n_linear

    # ---- quantize_weight accuracy, per-row vs per-tensor -------------------
    w = torch.randn(512, 256) * 0.05
    q, s = quantize_weight(w, per_row=True)
    deq = (q.float() * s)
    rel = (deq - w).abs().max() / w.abs().max()
    print(f"per-row round-trip rel err: {rel:.3e}")
    assert rel < 5e-2, rel
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert s.shape == (512, 1), s.shape

    # ---- structural conversion -------------------------------------------
    report = quantize_linears_(m, cache_dequant=False, min_params=0)
    print(f"converted={report['converted']} skipped={report['skipped']} "
          f"{report['orig_bytes']/2**20:.2f} -> {report['fp8_bytes']/2**20:.2f} MiB")
    assert report["converted"] == 26, report["converted"]
    assert not any(isinstance(x, nn.Linear) for x in m.modules()), "a nn.Linear survived"

    # nn.ModuleList container must still work (to_out[0])
    blk = m.blocks[0]
    assert isinstance(blk.attn.to_out[0], Fp8Weight), type(blk.attn.to_out[0])
    assert isinstance(blk.attn.to_out, nn.ModuleList)

    sm = fp8_memory_summary(m)
    print(f"summary: {sm}")
    assert sm["fp8_GiB"] < orig_bytes / 2**30, "fp8 did not shrink the weights"

    # ---- forward still runs ----------------------------------------------
    x = torch.randn(4, 64)
    with torch.no_grad():
        out = m(x)
    print(f"forward ok: out={tuple(out.shape)} sum={out.sum().item():.4f}")
    assert out.shape == (4, 64)

    # ---- .to(device) moves the custom buffers ----------------------------
    m2 = MiniTransformer(dim=dim, depth=1).eval()
    quantize_linears_(m2, min_params=0)
    before = m2.blocks[0].ff.proj.qweight.clone()
    m2 = m2.to(torch.bfloat16)          # dtype move
    w = m2.blocks[0].ff.proj
    assert w.qweight.dtype == torch.uint8, f"fp8 bytes clobbered: {w.qweight.dtype}"
    assert w.qweight.view(torch.float8_e4m3fn).dtype == torch.float8_e4m3fn
    assert w.scale.dtype == torch.float32, f"scale dtype was clobbered: {w.scale.dtype}"
    assert torch.equal(w.qweight, before), "storage bytes changed across .to(dtype)"
    with torch.no_grad():
        o2 = m2(torch.randn(4, 64, dtype=torch.bfloat16))
    assert o2.dtype == torch.bfloat16, o2.dtype
    print(f"dtype-move ok: fp8 bytes + fp32 scale preserved, out dtype {o2.dtype}")

    # `torch.cuda.is_available()` is not enough to decide this: it returns True even
    # when the dxg host-memory pool is exhausted and allocation fails, and an actual
    # failed allocation would poison this process's caching allocator. So probe in a
    # subprocess and skip the GPU leg cleanly when the driver cannot allocate.
    if gpu_alloc_works():
        m2 = m2.to("cuda")
        w = m2.blocks[0].ff.proj
        assert w.qweight.is_cuda, "fp8 weight did not move to GPU"
        assert w.qweight.dtype == torch.uint8
        assert w.dequantize().device.type == "cuda"
        with torch.no_grad():
            torch.cuda.synchronize()
            o3 = m2(torch.randn(4, 64, dtype=torch.bfloat16, device="cuda"))
            torch.cuda.synchronize()
        print(f"cuda-move ok: out={o3.sum().item():.4f} on {o3.device}")
    else:
        print("cuda-move SKIPPED: GPU cannot allocate (dxg pool exhausted)")
        print("  see GPU_RECOVERY_NEEDED.md; this is an environment state, not a code defect")

    print("\nALL FP8 CHECKS PASSED")


if __name__ == "__main__":
    main()
