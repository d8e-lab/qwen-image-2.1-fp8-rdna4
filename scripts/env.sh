#!/usr/bin/env bash
# Environment for Qwen-Image-2.1 on RX 9070 XT (gfx1201) / ROCm 7.14 / WSL2.
#
# Why an overlay instead of a plain venv:
#   - torch 2.12.0+rocm7.14.0 is only available as an AMD gfx1201 wheel (~3 GB) and is
#     already installed in torch_test/.venv. Re-downloading it over this flaky link is
#     wasteful, so we reuse that interpreter and shadow only the fast-moving packages.
#   - Qwen-Image-2.1 needs diffusers *main* (QwenImage21Pipeline is unreleased) which
#     requires huggingface-hub>=1.32, while the installed transformers 4.57.3 requires
#     huggingface-hub<1.0. The two cannot coexist in one site-packages, so newer
#     transformers + hub live in ./pylibs and come first on PYTHONPATH.
#
# Usage:  source scripts/env.sh && python scripts/run_qwen_image.py ...
set -u

QIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QIP_VENV="${QIP_VENV:-$HOME/workspace/torch_test/.venv}"
export QIP_PYLIBS="${QIP_PYLIBS:-$QIP_ROOT/pylibs}"
export QIP_MODEL="${QIP_MODEL:-$HOME/workspace/models/Qwen/Qwen-Image-2.1}"

# Overlay must precede site-packages so the newer transformers/hub win.
export PYTHONPATH="$QIP_ROOT/diffusers-src/src:$QIP_PYLIBS${PYTHONPATH:+:$PYTHONPATH}"

# Single dGPU here (RX 9070 XT on gfx1201). If a machine ever exposes more devices,
# pin by UUID via ROCR_VISIBLE_DEVICES. We deliberately do NOT hardcode a UUID.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1201}"

# hipBLASLt's gfx1201 Tensile kernels fail to load under WSL2/librocDXG ("named symbol not
# found" / "no kernel image is available"), and each failed lookup retries first. They are
# BYPASSED, never repaired - the failures still appear in logs during normal runs.
#
# Do not mistake this flag for the performance fix. Measured, with a contiguous (in, out)
# weight it changes nothing (122.6 vs 122.9 TFLOP/s). The real factors are:
#   (a) weight LAYOUT - a transposed-view weight runs at 0.1-0.4 TFLOP/s vs 99-123 for a
#       contiguous one, a ~1000x difference. Fp8Weight avoids it via `x @ Wt`.
#   (b) the SDPA kernel - pinning FLASH_ATTENTION avoids the MATH kernel's 4.91 GiB score
#       matrix (vs 0.06 GiB), which otherwise OOMs at 1024x1024.
# See NOTES.md sections 25, 26 and 31.
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
# rocBLAS can itself dispatch to hipBLASLt. Measured on this machine, disabling that
# halves the cost of the degenerate transposed-weight path (1943 ms -> 992 ms), and
# for normal contiguous weights the two paths are equivalent anyway (100.4 vs 97.9
# TFLOP/s), so there is no downside. ROCm issue #6203 reports the same symptom on
# gfx1201 and recommends this switch.
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"

# Attention backend. diffusers defaults to "native", which routes to
# torch.nn.functional.scaled_dot_product_attention. Leaving PyTorch to *choose* the SDPA
# kernel is not enough on this stack: for 4096-token bf16 causal attention it picks the
# MATH backend, which materialises the full score matrix and OOMs at 1024x1024.
# Measured peak allocation for one (1, 32, 4096, 128) causal bf16 attention:
#
#     FLASH_ATTENTION       0.06 GiB
#     EFFICIENT_ATTENTION   0.12 GiB
#     MATH                  4.91 GiB   <- what gets selected by default
#     CUDNN_ATTENTION       no kernel
#
# Forcing flash takes the 1024x1024 denoise step from OOM to comfortable, so it is pinned
# here. The QwenImage21FlexAttnProcessor alternative needs a compiled model and otherwise
# also materialises a dense score matrix, so "native" + explicit flash is the right pairing.
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-native}"
export TORCH_CUDA_ATTENTION_BACKEND="${TORCH_CUDA_ATTENTION_BACKEND:-FLASH_ATTENTION}"

# DO NOT set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True here.
# It was set in an earlier revision and it BREAKS this ROCm build: every CUDA
# allocation then fails with `hipErrorInvalidValue`, including an 8-byte tensor,
# which looks exactly like an exhausted GPU memory pool. Measured A/B on this
# machine, same interpreter and driver:
#     clean env                              -> OK
#     PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -> hipErrorInvalidValue
# The env var is intentionally left unset unless the caller provides one.

# Quiet the ROCm kernel-loading spam that otherwise drowns stdout.
export AMD_LOG_LEVEL=0
export HIP_LAUNCH_BLOCKING=0

# HuggingFace: model is already local, never phone home mid-run.
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1

# This machine's no_proxy lists a *bracketed* IPv6 loopback ("[::1]"). The httpx
# version in ./pylibs parses no_proxy entries as URLs and rejects the bracket,
# raising `httpx.InvalidURL: Invalid port: ':1]'` from deep inside huggingface_hub
# for every Hub call (load_config, from_pretrained, ...). Strip the brackets.
if [ -n "${no_proxy:-}" ]; then
  export no_proxy="${no_proxy//\[/}"; export no_proxy="${no_proxy//\]/}"
fi
if [ -n "${NO_PROXY:-}" ]; then
  export NO_PROXY="${NO_PROXY//\[/}"; export NO_PROXY="${NO_PROXY//\]/}"
fi

PY="$QIP_VENV/bin/python"

qip_python() { "$PY" "$@"; }

qip_check() {
  "$PY" - <<'EOF'
import sys, torch, transformers, diffusers
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| hip", torch.version.hip)
print("transformers", transformers.__version__, "from", transformers.__file__)
print("diffusers   ", diffusers.__version__, "from", diffusers.__file__)
print("gpu         ", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print("vram free   ", round(free / 2**30, 3), "GiB /", round(total / 2**30, 3), "GiB")
EOF
}

# Detect the exhausted-dxg-pool state described in NOTES.md section 2.1. A failed
# HIP allocation poisons the in-process caching allocator, so the probe must run in
# a throwaway subprocess, never in the caller's.
qip_gpu_ok() {
  "$PY" - <<'EOF'
import subprocess, sys
code = ("import torch;torch.cuda.init();"
        "t=torch.zeros(8,dtype=torch.bfloat16,device='cuda');"
        "torch.cuda.synchronize();print('OK')")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
if r.returncode == 0 and "OK" in r.stdout:
    print("GPU OK - allocations work")
    raise SystemExit(0)
print("GPU UNUSABLE - cannot allocate 8 bytes.")
print("  Check, in order:")
print("   1. PYTORCH_HIP_ALLOC_CONF must NOT contain expandable_segments "
      "(breaks this ROCm build).")
print("   2. If that is unset and it still fails, reset the AMD adapter on Windows")
print("      (Restart-GPU.ps1 as Administrator, or reboot).")
if r.stderr:
    tail = [l for l in r.stderr.strip().splitlines() if l.strip()][-3:]
    for l in tail:
        print("   ", l.strip()[:160])
raise SystemExit(1)
EOF
}
