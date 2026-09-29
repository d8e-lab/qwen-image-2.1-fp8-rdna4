"""FP8 weight-only quantization for the Qwen-Image-2.1 transformer on ROCm/RDNA4.

Why weight-only and not real FP8 GEMM
-------------------------------------
The RX 9070 XT (gfx1201, RDNA4) does have FP8 WMMA hardware, and this ROCm build
even ships hipBLASLt FP8 kernels (``Type_F8F8``/``Type_F8H`` for gfx1201). They are
unusable here: under WSL2/librocDXG every ``hipModuleLoad`` of those Tensile code
objects fails with "named symbol not found" / "no kernel image is available for
execution on the device", so ``torch._scaled_mm`` raises
``HIPBLAS_STATUS_INTERNAL_ERROR`` for e4m3fn and ``HIPBLAS_STATUS_NOT_SUPPORTED``
for e4m3fnuz. Full half-precision *compute* is numerically fine on this stack
(measured rel. error 4e-4 for fp16, 3.7e-3 for bf16 vs fp64), so we keep bf16
math and use FP8 purely as a *storage* format.

What this buys
--------------
The transformer is 7.1B parameters: 14.27 GB in bf16, which does not fit in this
card's 15.8 GB alongside activations, the Qwen3-VL text encoder, and the VAE.
Stored as float8_e4m3fn it is ~7.1 GB, which does.

Cost: the weight must be materialised back to bf16 for each matmul. ``Fp8Weight``
optionally caches that dequantised copy so a hot layer pays the cost once.

Scaling
-------
Per-output-row scaling (one scale per row of the weight) is used because it is a
strict accuracy win over a single per-tensor scale for essentially no run-time cost:
the dequant is a broadcast multiply. Scales are stored fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Largest magnitude representable by float8_e4m3fn.
FP8_MAX = 448.0

# torch renamed/added fp8 dtypes over several releases; pick the best available.
_E4M3 = getattr(torch, "float8_e4m3fn", None)
if _E4M3 is None:  # pragma: no cover - very old torch
    raise RuntimeError("This torch build has no torch.float8_e4m3fn")


def quantize_weight(
    weight: torch.Tensor,
    per_row: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to fp8, returning (qweight, scale).

    With ``per_row`` the scale has shape (out_features, 1) so that
    ``qweight.float() * scale`` broadcasts back to the original layout.
    """
    w = weight.detach().float()
    if per_row:
        amax = w.abs().amax(dim=1, keepdim=True)
    else:
        amax = w.abs().amax()
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(_E4M3)
    return q, scale


class Fp8Weight(nn.Module):
    """Holds an fp8 weight plus its scale; materialises bf16 on demand.

    Deliberately not an ``nn.Linear`` subclass so that nothing can accidentally
    reach the original fp32/bf16 ``.weight`` and blow the memory budget back up.

    Storage is ``torch.uint8`` holding the raw fp8 bytes rather than a
    ``torch.float8_e4m3fn`` tensor. On this ROCm build a float8 tensor cannot be
    copied to the GPU at all (``.to('cuda')`` raises ``hipErrorInvalidValue``,
    and the failed copy leaves the caching allocator asserting), whereas uint8
    moves fine and ``view(torch.float8_e4m3fn)`` reinterprets it on-device.
    Keeping bytes also means ``nn.Module.to(dtype=...)`` cannot silently upcast
    the storage, since there is no floating dtype to convert.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        cache_dequant: bool = False,
    ) -> None:
        super().__init__()
        # Accept either dtype; always store raw bytes.
        if qweight.dtype == _E4M3:
            qweight = qweight.view(torch.uint8)
        elif qweight.dtype != torch.uint8:
            raise TypeError(f"expected float8_e4m3fn or uint8 bytes, got {qweight.dtype}")
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        self.cache_dequant = cache_dequant
        self._cache: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    @property
    def in_features(self) -> int:
        return self.qweight.shape[1]

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return the weight materialised in ``dtype`` (bf16 by default), shape (out, in)."""
        return self.dequantize_transposed(dtype).t()

    def dequantize_transposed(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return the weight materialised as a CONTIGUOUS (in, out) tensor.

        Layout matters enormously here. On this ROCm build a linear whose weight is a
        *transposed view* falls into a degenerate path - hipBLASLt cannot load the
        kernel for that combination and it retries then falls back to something
        thousands of times slower. Measured on the RX 9070 XT, 1024x4096 @ 12288:

            x @ W.t()   (W is (out, in))     1010 ms    0.10 TFLOP/s
            x @ Wt      (Wt contiguous (in,out)) 1.22 ms   84.3 TFLOP/s

        and `F.linear(x, W)` / `nn.Linear` hit the same slow path because they
        internally transpose. Since every linear in this model would pay that cost,
        the weight is materialised transposed and used via `x @ Wt`.

        The fp32 multiply applies the per-row scale at full precision before the
        single narrowing cast; `q.float()` converts in-register (it is not a bit
        reinterpretation), which is what dequantisation needs.
        """
        if self._cache is not None and self._cache.dtype == dtype:
            return self._cache
        q = self.qweight.view(_E4M3)
        # (out, in) -> transpose -> contiguous (in, out)
        w = (q.float() * self.scale).to(dtype).t().contiguous()
        if self.cache_dequant:
            self._cache = w
        return w

    def drop_cache(self) -> None:
        self._cache = None

    def _apply(self, fn, recurse: bool = True):
        """Device moves only; never let a dtype cast touch the storage.

        ``nn.Module.to(torch.bfloat16)`` would rewrite ``scale`` from fp32 to
        bfloat16 (torch 2.12 does not exempt fp8 buffers the way older releases
        did) and silently change the quantisation. So instead of delegating to
        ``fn`` we move the tensors and leave their dtypes alone. Returns ``self``
        like the base implementation.
        """
        self.qweight = self.qweight.to(device=fn(self.qweight).device)
        self.scale = self.scale.to(device=self.qweight.device)
        if self.bias is not None:
            self.bias = self.bias.to(device=self.qweight.device)
        self._cache = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror the input dtype: the real pipeline runs bf16 throughout, but this
        # keeps the module usable if a caller hands it fp32/fp16 activations.
        dt = x.dtype if x.is_floating_point() else torch.bfloat16
        w_t = self.dequantize_transposed(dt)          # contiguous (in, out)
        out = x @ w_t                                  # fast path; see dequantize_transposed
        if self.bias is not None:
            b = self.bias if self.bias.dtype == out.dtype else self.bias.to(out.dtype)
            out = out + b
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"fp8_e4m3fn, per_row_scale, cache={self.cache_dequant}"
        )


def quantize_linears_(
    module: nn.Module,
    cache_dequant: bool = False,
    skip_names: tuple[str, ...] = (),
    min_params: int = 1 << 16,
) -> dict:
    """Replace every ``nn.Linear`` in ``module`` with an :class:`Fp8Weight` in place.

    ``min_params`` skips tiny projections (e.g. the 4*dim modulation MLP input)
    where fp8 saves nothing meaningful but adds a kernel per call.

    The original fp32/bf16 parameter is dropped from the module as each layer is
    converted, so peak memory stays close to the fp8 size rather than requiring a
    full second copy. Returns a small report.
    """
    report = {"converted": 0, "skipped": 0, "fp8_bytes": 0, "orig_bytes": 0, "layers": []}

    def _walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(s in full for s in skip_names) or child.weight.numel() < min_params:
                    report["skipped"] += 1
                    continue
                w = child.weight.data
                q, scale = quantize_weight(w, per_row=True)
                bias = child.bias.data if child.bias is not None else None
                new = Fp8Weight(q, scale, bias, cache_dequant=cache_dequant)
                new = new.to(device=w.device)
                report["fp8_bytes"] += q.numel() * q.element_size() + scale.numel() * 4
                report["orig_bytes"] += w.numel() * w.element_size()
                report["layers"].append(full)
                report["converted"] += 1
                setattr(parent, name, new)
                # Drop the bf16/fp32 source immediately.
                del child, w
            else:
                _walk(child, full)

    _walk(module, "")
    return report


def fp8_memory_summary(model: nn.Module) -> dict:
    """Report where the bytes went, split fp8 vs everything else."""
    fp8_bytes = other_bytes = 0
    n_fp8 = n_other = 0
    for m in model.modules():
        if isinstance(m, Fp8Weight):
            fp8_bytes += m.qweight.numel() * m.qweight.element_size()
            fp8_bytes += m.scale.numel() * m.scale.element_size()
            if m.bias is not None:
                fp8_bytes += m.bias.numel() * m.bias.element_size()
            n_fp8 += 1
        else:
            for p in m.parameters(recurse=False):
                other_bytes += p.numel() * p.element_size()
                n_other += 1
            for b in m.buffers(recurse=False):
                other_bytes += b.numel() * b.element_size()
    return {
        "fp8_layers": n_fp8,
        "fp8_GiB": fp8_bytes / 2**30,
        "other_params": n_other,
        "other_GiB": other_bytes / 2**30,
        "total_GiB": (fp8_bytes + other_bytes) / 2**30,
    }


def gpu_alloc_works(timeout: int = 180) -> bool:
    """True only if the GPU can actually allocate memory.

    ``torch.cuda.is_available()`` is not sufficient: on this machine it returns True
    even when the dxg host-memory pool is exhausted and every allocation fails.

    The probe must run in a subprocess. A failed HIP allocation leaves the calling
    process's caching allocator asserting
    (``HIPCachingAllocator ... !handles_.at(i) INTERNAL ASSERT FAILED``), which would
    poison every later test in the same interpreter.
    """
    import os
    import subprocess
    import sys

    code = (
        "import torch;torch.cuda.init();"
        "t=torch.zeros(8,dtype=torch.bfloat16,device='cuda');"
        "torch.cuda.synchronize();print('OK')"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout, env=dict(os.environ))
    except Exception:
        return False
    return r.returncode == 0 and "OK" in r.stdout
