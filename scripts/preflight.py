#!/usr/bin/env python3
"""Preflight: is everything needed for a real Qwen-Image-2.1 GPU run in place?

Run this before `run_qwen_image.py`. It answers, without loading any weights:
  * is the environment (overlay + torch) wired up?
  * is the download complete, including the zero-byte config files that signal a
    finished `snapshot_download` (a partially downloaded repo has the weights but may
    still be missing `vae/config.json`)?
  * is the GPU actually usable, given the dxg pool can be left exhausted?

Exit code 0 means "ready"; non-zero lists what is missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

GIB = 2**30

# Expected byte sizes for the large files, from the Hub manifest. A size mismatch means
# an interrupted download; huggingface_hub leaves those as .incomplete blobs.
EXPECTED = {
    "text_encoder/model-00001-of-00004.safetensors": 4998056552,
    "text_encoder/model-00002-of-00004.safetensors": 4915962464,
    "text_encoder/model-00003-of-00004.safetensors": 4915962496,
    "text_encoder/model-00004-of-00004.safetensors": 2704357976,
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 9968332504,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 4261951904,
    "vae/diffusion_pytorch_model.safetensors": 1350989512,
}
REQUIRED_SMALL = [
    "model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
    "processor/tokenizer.json",
    "processor/preprocessor_config.json",
]


def main() -> int:
    model = os.environ.get("QIP_MODEL", os.path.expanduser(
        "~/workspace/models/Qwen/Qwen-Image-2.1"))
    problems: list[str] = []
    notes: list[str] = []

    print("=" * 70)
    print("Qwen-Image-2.1 preflight")
    print("=" * 70)

    # ---- environment --------------------------------------------------------
    print("\n[environment]")
    try:
        import torch
        import transformers
        import diffusers
        from diffusers import QwenImage21Pipeline  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"  FAIL importing the stack: {type(e).__name__}: {e}")
        print("  -> did you `source scripts/env.sh`?")
        return 2
    print(f"  torch        {torch.__version__} (hip {torch.version.hip})")
    print(f"  transformers {transformers.__version__}")
    print(f"  diffusers    {diffusers.__version__}")
    if "diffusers-src" not in diffusers.__file__:
        problems.append("diffusers is not the git-main overlay (need QwenImage21Pipeline)")
    if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") != "0":
        problems.append(
            "TORCH_BLAS_PREFER_HIPBLASLT != 0 - hipBLASLt leaks the dxg GPU pool; "
            "source scripts/env.sh"
        )
    else:
        notes.append("hipBLASLt disabled (required)")

    # ---- download -----------------------------------------------------------
    print("\n[download]")
    if not os.path.isdir(model):
        problems.append(f"model directory missing: {model}")
    else:
        total = done = 0
        for rel, want in EXPECTED.items():
            p = os.path.join(model, rel)
            # fast_download.py stages into <name>.part, which must not be mistaken
            # for the finished file just because it shares the extension.
            part = p + ".part"
            if os.path.isfile(part) and not os.path.isfile(p):
                notes.append(f"in progress via fast_download: {rel} "
                             f"({os.path.getsize(part)/1e9:.2f} GB staged)")
            size = os.path.getsize(p) if os.path.isfile(p) else 0
            total += want
            done += min(size, want)
            if size == 0:
                problems.append(f"missing: {rel}")
            elif size < want * 0.995:
                problems.append(f"incomplete: {rel} ({size/1e9:.2f} of {want/1e9:.2f} GB)")
        print(f"  weights  {done/1e9:.2f} / {total/1e9:.2f} GB ({done/total*100:.1f}%) finished")

        # Count in-flight bytes from both mechanisms, without double counting. The
        # .incomplete blobs are what snapshot_download leaves behind; .part files are
        # fast_download.py's staging. Only one of the two is active at a time, and the
        # .part files supersede the blobs for the same target, so take the larger of
        # the two figures per logical target rather than summing them.
        cache = os.path.join(model, ".cache", "huggingface", "download")
        blob_bytes = 0
        n_inc = 0
        if os.path.isdir(cache):
            for dp, _, fs in os.walk(cache):
                for f in fs:
                    if f.endswith(".incomplete"):
                        n_inc += 1
                        try:
                            blob_bytes += os.path.getsize(os.path.join(dp, f))
                        except OSError:
                            pass
        part_bytes = 0
        n_part = 0
        for dp, _, fs in os.walk(model):
            if os.sep + ".cache" in dp:
                continue
            for f in fs:
                if f.endswith(".part"):
                    n_part += 1
                    try:
                        part_bytes += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
        # Progress must be computed from finished bytes plus *live* staging only.
        # The .incomplete blobs are leftovers from snapshot_download; they are seed
        # material that fast_download may reuse, not remaining work, so adding them
        # produced a nonsensical >100% figure. Report them separately.
        inflight = part_bytes
        if inflight:
            overall = min(100.0, (done + inflight) / total * 100)
            print(f"  in-flight {inflight/1e9:.2f} GB in {n_part} .part file(s)")
            print(f"           -> {overall:.1f}% of total bytes fetched")
        elif n_inc:
            print(f"  in-flight 0.00 GB (no .part staging)")
        if blob_bytes:
            print(f"  leftover .incomplete blobs: {blob_bytes/1e9:.2f} GB "
                  f"(seed material from snapshot_download; safe to delete to reclaim disk)")
        for rel in REQUIRED_SMALL:
            if not os.path.isfile(os.path.join(model, rel)):
                problems.append(f"missing: {rel}")

        if n_inc:
            notes.append(f"{n_inc} leftover .incomplete blob(s), {blob_bytes/1e9:.2f} GB "
                         f"(seed material; deletable)")
        if n_part:
            notes.append(f"{n_part} .part file(s) staged by fast_download (resumable)")

    # ---- GPU ----------------------------------------------------------------
    print("\n[gpu]")
    code = (
        "import torch;torch.cuda.init();"
        "t=torch.zeros(8,dtype=torch.bfloat16,device='cuda');"
        "torch.cuda.synchronize();print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    if r.returncode == 0 and "OK" in r.stdout:
        print("  allocations work")
        notes.append("GPU usable")
    else:
        print("  GPU CANNOT allocate 8 bytes")
        problems.append(
            "GPU unusable: leaked dxg host-memory pool. Restart the AMD display adapter "
            "on Windows (Device Manager -> disable/enable) or reboot. See GPU_RECOVERY_NEEDED.md"
        )

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 70)
    if notes:
        print("notes:")
        for n in notes:
            print(f"  - {n}")
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  x {p}")
        print("=" * 70)
        return 1
    print("READY - everything is in place for a real run.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
