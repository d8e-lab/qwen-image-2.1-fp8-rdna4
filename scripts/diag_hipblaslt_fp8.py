#!/usr/bin/env python3
"""Is hipBLASLt's fp8 path usable on this machine? One script, one table.

Written to be run unchanged on WSL2 and on native Linux so the two can be compared
directly - the open question in NOTES.md section 39 is whether the CCOB Tensile
libraries fail to load only under WSL (`librocDXG`) or everywhere.

What it checks
==============
1. device + stack identity (name, gcnArchName, ROCm/torch versions, the two env switches)
2. `torch._scaled_mm` fp8 e4m3fn / e5m2 at several shapes - the operation that fails
3. bf16 matmul, run twice: with rocBLAS (the `ROCBLAS_USE_HIPBLASLT=0` default) and with
   hipBLASLt forced, because hipBLASLt's bf16 path also loads CCOB libraries and silently
   falls back (that fallback is the ~1000x slowdown in NOTES.md section 25)
4. `torch._int_mm` int8 - the low-precision path that does work, as a control

`--tensile-db` re-runs the fp8 probe in a child process with `TENSILE_DB=0xffff` and
reports how many `hipModuleLoad failed` / `no kernel image` lines the loader emitted,
which is the raw evidence an upstream report needs.

Usage
=====
    $QIP_VENV/bin/python scripts/diag_hipblaslt_fp8.py
    $QIP_VENV/bin/python scripts/diag_hipblaslt_fp8.py --tensile-db
    $QIP_VENV/bin/python scripts/diag_hipblaslt_fp8.py --json-out diag.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch


def env_facts() -> dict:
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device": p.name if p else None,
        "gcnArchName": getattr(p, "gcnArchName", None) if p else None,
        "ROCBLAS_USE_HIPBLASLT": os.environ.get("ROCBLAS_USE_HIPBLASLT", "<unset>"),
        "TORCH_BLAS_PREFER_HIPBLASLT": os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "<unset>"),
        "python": sys.version.split()[0],
    }


def timeit(fn, n: int = 5) -> tuple[bool, float | str]:
    try:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return True, (time.perf_counter() - t0) / n * 1000
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


def probe_scaled(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[bool, float | str]:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(dtype)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(dtype)
    s = torch.tensor([1.0], device="cuda")
    return timeit(lambda: torch._scaled_mm(a, b.t(), scale_a=s, scale_b=s,
                                           out_dtype=torch.bfloat16))


def probe_int8(M: int, N: int, K: int) -> tuple[bool, float | str]:
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
    return timeit(lambda: torch._int_mm(a, b))


def probe_bf16(M: int, N: int, K: int) -> tuple[bool, float | str]:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t().contiguous()
    return timeit(lambda: a @ w)


def tensile_db_run() -> dict:
    """Re-run the fp8 probe under TENSILE_DB and count the loader complaints."""
    code = (
        "import torch\n"
        "a=torch.randn(128,512,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)\n"
        "b=torch.randn(128,512,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)\n"
        "s=torch.tensor([1.0],device='cuda')\n"
        "try:\n"
        "    torch._scaled_mm(a,b.t(),scale_a=s,scale_b=s,out_dtype=torch.bfloat16)\n"
        "    torch.cuda.synchronize(); print('FP8_OK')\n"
        "except Exception as e:\n"
        "    print('FP8_FAIL', str(e).splitlines()[0][:120])\n"
    )
    env = dict(os.environ, TENSILE_DB="0xffff")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, timeout=600)
    err = r.stderr
    verdict = [ln for ln in r.stdout.splitlines() if ln.startswith("FP8_")]
    return {
        "verdict": verdict[0] if verdict else "",
        "module_load_failed": err.count("hipModuleLoad failed"),
        "no_kernel_image": err.count("no kernel image is available"),
        "named_symbol": err.count("named symbol not found"),
        "loaded_hsaco": (r.stdout + err).count("loaded code object"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensile-db", action="store_true", help="also collect loader counts")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    facts = env_facts()
    print("=" * 78)
    for k, v in facts.items():
        print(f"  {k:26s} {v}")
    if not facts["cuda_available"]:
        print("no GPU visible - nothing to probe")
        return 1

    results: dict[str, list] = {}
    print("\n-- fp8 e4m3fn (`torch._scaled_mm`) --")
    results["e4m3"] = []
    for M, N, K in [(64, 256, 512), (128, 128, 512), (1024, 1024, 1024), (4096, 4096, 4096)]:
        ok, val = probe_scaled(M, N, K, torch.float8_e4m3fn)
        results["e4m3"].append([M, N, K, ok, val])
        print(f"  {M:5d}x{N:5d}x{K:5d}  {'OK  %8.2f ms' % val if ok else 'FAIL  ' + val}")

    print("\n-- fp8 e5m2 --")
    ok, val = probe_scaled(128, 128, 512, torch.float8_e5m2)
    results["e5m2"] = [[128, 128, 512, ok, val]]
    print(f"  128x128x512      {'OK  %8.2f ms' % val if ok else 'FAIL  ' + val}")

    print("\n-- bf16 (control: rocBLAS vs forced hipBLASLt) --")
    ok, val = probe_bf16(1024, 1024, 1024)
    results["bf16_rocblas"] = [ok, val]
    print(f"  rocBLAS default  {'OK  %8.2f ms' % val if ok else 'FAIL  ' + val}")
    env_note = ("(re-run with ROCBLAS_USE_HIPBLASLT=1 to compare the hipBLASLt path, and with "
                "TENSILE_DB=0xffff to see the CCOB load failures)")
    print("  " + env_note)

    print("\n-- int8 (`torch._int_mm`, control) --")
    ok8, val8 = probe_int8(1024, 1024, 1024)
    results["int8"] = [ok8, val8]
    print(f"  1024x1024x1024   {'OK  %8.2f ms' % val8 if ok8 else 'FAIL  ' + val8}")

    if args.tensile_db:
        print("\n-- loader counts under TENSILE_DB=0xffff --")
        db = tensile_db_run()
        for k, v in db.items():
            print(f"  {k:20s} {v}")
        results["tensile_db"] = db

    print("\n" + "=" * 78)
    fp8_works = any(r[3] for r in results["e4m3"])
    print(f"  fp8 GEMM: {'AVAILABLE' if fp8_works else 'UNAVAILABLE'}"
          f"   |  int8 GEMM: {'AVAILABLE' if ok8 else 'UNAVAILABLE'}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"env": facts, "results": results}, fh, indent=2)
        print(f"  wrote {args.json_out}")
    return 0 if fp8_works else 2


if __name__ == "__main__":
    raise SystemExit(main())
