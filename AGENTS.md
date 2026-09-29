# Working in this directory: Qwen-Image-2.1 fp8 deployment

Read `NOTES.md` first — it records the full environment, every non-obvious finding, and the
status checklist. Do not rediscover these the hard way.

## Environment

Always work through the overlay environment; the system python has no torch:

```bash
cd ~/workspace/qwen-image
source scripts/env.sh          # sets PYTHONPATH overlay + TORCH_BLAS_PREFER_HIPBLASLT=0
qip_check                      # versions + VRAM
qip_gpu_ok                     # is the GPU actually usable right now?
$QIP_VENV/bin/python scripts/...   # run things with this interpreter
```

## Hard rules (each one was learned from a failure)

1. **The layout of a linear weight is the dominant performance factor, by ~1000x.**
   A transposed-view weight runs at 0.1-0.4 TFLOP/s; a contiguous `(in, out)` weight at
   99-123 TFLOP/s. `F.linear` and `nn.Linear` both hit the slow path because they transpose
   internally. `Fp8Weight` materialises the weight transposed and uses `x @ Wt`; do not
   "simplify" that back to `F.linear`.

   `TORCH_BLAS_PREFER_HIPBLASLT=0` is kept set, but do **not** mistake it for the fix: with a
   contiguous weight it changes nothing measurable (122.6 vs 122.9 TFLOP/s). hipBLASLt's
   gfx1201 Tensile kernels genuinely fail to load under WSL2/librocDXG and are **bypassed,
   never repaired** - the failures still appear in logs during ordinary runs. The flag merely
   avoids the failed-lookup retry churn.

   **Never set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.** An earlier revision did,
   and it breaks allocation on this ROCm build — every tensor allocation fails with
   `hipErrorInvalidValue`, including 8 bytes, which looks exactly like an exhausted GPU memory
   pool. That mistake cost ten rounds of misdiagnosis and an unnecessary Windows reboot
   (NOTES.md section 24). If the GPU ever reports UNUSABLE, check this variable first.
2. **Never allocate and fail on the GPU in a long-lived process.** A failed HIP allocation
   leaves the caching allocator asserting. Probe with `qip_gpu_ok`, which uses a subprocess.
3. **Store fp8 weights as `uint8` bytes, not `float8_e4m3fn`.** A float8 tensor cannot be copied
   to this GPU at all (`hipErrorInvalidValue`). `Fp8Weight` keeps bytes and `.view()`s them
   on-device. Do not "simplify" that away.
4. **Never call `.to(dtype)` on a model containing `Fp8Weight`.** `Fp8Weight._apply` deliberately
   only moves devices; a dtype cast would rewrite the fp8 bytes and the fp32 scale.
5. **VAE tiling is on by default for GPU >=768px.** The VAE *decode* is the peak allocation
   in the whole run, not the denoise loop. Call `pipe.vae.enable_tiling()` — `QwenImage21Pipeline`
   does not forward `enable_vae_tiling()`.
6. **Prompt encoding runs on CPU even in a GPU run.** The fp8 transformer is resident in VRAM
   and the text encoder needs 16.33 GiB, so they cannot coexist. Encoding takes ~16 s on CPU
   against minutes of denoising.
7. **Respect the VRAM budget.** Roughly 14 of 15.82 GiB is free (the Windows desktop holds the
   rest) and an OOM on this box has taken the host down before. Measured peak across a
   1024x1024 run is ~10.3 GiB. Guards exist for this; do not lower them casually.
   `run_qwen_image.py` loads on CPU, quantises, then moves — keep that ordering.

   **When writing a memory guard, remember what the number means.** `mem_get_info().free` is
   what is *left*, not what is *total*: subtracting already-resident weights from it
   double-counts. This exact mistake appeared three times in this repo (NOTES.md sections 28,
   30) and each time silently disabled a feature rather than erroring.

8. **`run_qwen_image.py` is the entry point.** It never registers with `accelerate`
   (`enable_model_cpu_offload()` would bypass the fp8 weights through its own meta-device
   loading). Staged loading bounds memory: the transformer is quantised on CPU and moved once,
   the 16.33 GiB text encoder is freed before the VAE loads, and on GPU the prompt is encoded
   on CPU so the encoder and the resident transformer never compete for VRAM.

## Background work

Long jobs get killed with the shell. Launch them detached and poll a log file:

```bash
setsid nohup $QIP_VENV/bin/python -u scripts/download.py \
    > logs/download.log 2>&1 < /dev/null &
```

`scripts/resume_download.sh` does this for the model download and is safe to re-run (it resumes
from `.incomplete` shards).
