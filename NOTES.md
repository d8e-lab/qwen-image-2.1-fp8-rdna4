# Qwen-Image-2.1 on RX 9070 XT (gfx1201) / ROCm 7.14 / WSL2 — deployment notes

Target: run `Qwen/Qwen-Image-2.1` locally with **fp8-quantised** transformer weights on a
16 GB RDNA4 card, without ever hitting an OOM/blue-screen.

Host: Windows 10 + WSL2, Arch Linux, Ryzen 5 9600X, **Radeon RX 9070 XT (gfx1201, 15.82 GiB
visible VRAM, ~13.3 GiB actually free** — the Windows desktop holds the rest).
ROCm 7.14 (TheRock via AUR `rocm-gfx120x-bin`), `torch 2.12.0+rocm7.14.0`.

---

## 1. What the model actually is

From the Hub manifest and `transformer/config.json`:

| component | class | params | bf16 size |
|---|---|---|---|
| transformer | `QwenImage21Transformer2DModel` | **7.115 B** (measured) | **13.25 GiB** |
| text encoder | `Qwen3VLForConditionalGeneration` (Qwen3-VL 4B) | ~4.3 B | ~7.26 GiB |
| vae | `AutoencoderKLQwenImage21` | — | ~0.68 GiB |

Total download **33.13 GB**. The README's "7B parameters in its visual generation component
(32 Single-Stream DiT layers)" matches the measured 7.115 B exactly, so the *transformer* is
the 7B part — the text encoder is a separate ~4B Qwen3-VL.

`13.25 GiB` of weights does **not** fit next to activations, a KV cache, and the VAE in
13.3 GiB of free VRAM. That is the whole reason for fp8.

---

## 2. Findings that shaped the design

### 2.1 hipBLASLt is catastrophically broken on this stack (and leaks GPU memory)

This ROCm build ships gfx1201 Tensile kernels, but under WSL2/librocDXG **every**
`hipModuleLoad` of them fails:

```
hipModuleLoad failed: .../hipblaslt/library/gfx1201/TensileLibrary_..._gfx1201.co
 error: no kernel image is available for execution on the device
 error: named symbol not found
```

Each failure triggers `Clearing modules and retrying hipModuleLoad` and then a fallback to an
untuned rocBLAS path. The cost is not a few percent — it is total:

| op | hipBLASLt enabled | disabled |
|---|---|---|
| `64x1024x4096` bf16 | **1147 ms** | **0.049 ms** (23,000x) |
| `4096x4096x12288` bf16 | unusable | **3.2 ms** (128 TFLOP/s) |

**Fix:** `TORCH_BLAS_PREFER_HIPBLASLT=0` (see `scripts/env.sh`).

**Worse:** those repeated failed loads leak the WSL2 dxg GPU host-memory pool. After enough of
them the kernel logs thousands of

```
misc dxg: dxgk: create_existing_sysmem: establish_gpadl failed: -122
hv_vmbus: Failed to establish GPADL: err = 0xc0000004
```

and then **no allocation works at all**, not even an 8-byte tensor
(`hipErrorInvalidValue`, and `HIPCachingAllocator ... !handles_.at(i) INTERNAL ASSERT FAILED`).
The pool is not reclaimed when processes exit — **only a WSL restart clears it**. This is why
the work was interrupted once. Keeping the env var set avoids re-triggering it.

This is also the most likely explanation for the original "fp16/bf16 fails, fp32 works"
symptom on this machine: it matches the user's own observation that **native Linux half
precision is fine**. Half precision here is *numerically* fine (measured vs fp64:
rel. err **4e-4 fp16**, **3.7e-3 bf16**) — the failures come from the kernel-loading path.

### 2.2 Native FP8 GEMM is unavailable

`torch._scaled_mm` fails for every fp8 dtype:

* `float8_e4m3fn` → `HIPBLAS_STATUS_INTERNAL_ERROR` (hipblasLtMatmul, computeType 2)
* `float8_e4m3fnuz` → `HIPBLAS_STATUS_NOT_SUPPORTED`
* `float8_e5m2` → `HIPBLAS_STATUS_INTERNAL_ERROR`

…even though the kernels are present on disk (`Type_F8F8`, `Type_F8H` for gfx1201). So **real
fp8 matmul is off the table**; fp8 is used purely as a *storage* format with bf16 math.

### 2.3 A float8 tensor cannot be copied to the GPU

```python
torch.randn(256, 128).to(torch.float8_e4m3fn).to("cuda")
# AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
```

and the failed copy leaves the allocator asserting. `uint8` moves fine, so
`Fp8Weight` stores **raw bytes as `uint8`** and reinterprets them on-device with
`.view(torch.float8_e4m3fn)` (verified working). Bonus: because there is no floating dtype in
storage, `nn.Module.to(dtype=...)` cannot silently upcast the weights — which it otherwise does
on torch 2.12, since `_apply` no longer exempts float8 buffers.

### 2.4 torch 2.12 `.to(dtype)` clobbers fp8/scale buffers

`model.to(torch.bfloat16)` rewrote `qweight` (fp8→bf16) and `scale` (fp32→bf16), undoing the
quantisation. `Fp8Weight._apply` now performs device moves only and never dtype casts.

### 2.5 Attention is fine — the default backend is the safe one

The diffusers source warns that `QwenImage21FlexAttnProcessor` without `torch.compile` falls
back to a dense fp32 score matrix and OOMs at high resolution. That is **not** the default:
`DIFFUSERS_ATTN_BACKEND` defaults to `native`, which routes to
`torch.nn.functional.scaled_dot_product_attention` (memory-efficient/flash kernels).
Verified by reading `attention_dispatch.py`. Keep `native`.

---

## 3. Measured fp8 result (real transformer shape, CPU)

From `scripts/test_fp8_real.py`:

```
parameters : 7.115 B
fp32 bytes : 26.506 GiB
bf16 bytes : 13.253 GiB   <- does not fit in 13.3 GiB free
nn.Linear  : 232

quantised 232 layers: 13.253 GiB -> 6.632 GiB (2.00x)
transformer total after fp8: 6.632 GiB  (saved 6.621 GiB)
```

Quantisation is **per-output-row** scaling, which measurably beats a single per-tensor scale
(rel. err 2.4e-2 vs 2.9e-2) for no runtime cost — dequant is a broadcast multiply.

### VRAM budget at 1024x1024

| item | GiB |
|---|---|
| transformer (fp8) | 6.63 |
| KV cache (32 layers x 2 x 4096 seq x 4096 d, bf16) | 2.00 |
| activations | ~1.0 |
| vae (bf16) | 0.68 |
| **total** | **~10.3** of 13.3 free |

The Qwen3-VL text encoder (7.26 GiB) is needed only *before* denoising, so it is loaded,
used, then freed — it never coexists with the KV cache. `--guidance > 1` doubles the KV cache
and runs the transformer twice per step, so it is off by default.

---

## 4. Environment layout

Qwen-Image-2.1 needs **diffusers from git main** (`QwenImage21Pipeline` is unreleased — not in
0.40.0) and **transformers >= 5.17**. But diffusers main requires `huggingface-hub>=1.32`,
while the installed `transformers 4.57.3` requires `hub<1.0`. They cannot share one
site-packages, and the ROCm torch wheel (~3 GB) should not be re-downloaded over this link.

So `scripts/env.sh` reuses the existing interpreter and puts an **overlay** first on
`PYTHONPATH`:

```
torch 2.12.0+rocm7.14.0   (reused from ~/workspace/torch_test/.venv)
transformers 5.17.0       (./pylibs)
huggingface_hub 1.33.0    (./pylibs)
diffusers 0.41.0.dev0     (./diffusers-src, git main)
```

Verify with `source scripts/env.sh && qip_check`.

---

## 5. Files

| file | purpose |
|---|---|
| `scripts/env.sh` | environment + the `TORCH_BLAS_PREFER_HIPBLASLT=0` fix; `qip_check` health check |
| `scripts/fp8_quant.py` | `Fp8Weight`, `quantize_linears_`, memory accounting |
| `scripts/run_qwen_image.py` | end-to-end generation with staged loading + VRAM guards |
| `scripts/test_fp8.py` | unit tests (toy module, 26 Linears) |
| `scripts/test_fp8_real.py` | validation on the real 7.115 B config |
| `scripts/bench_gemm.py` | the GEMM/hipBLASLt measurements above |
| `scripts/bench_fp8b.py` | fp8 vs bf16 timing + per-tensor vs per-row accuracy |
| `scripts/download.py` | resumable snapshot download (retries forever) |
| `scripts/watch_download.py` | appends progress to `logs/download_watch.log` |

---

## 6. Status / how to resume

- [x] Model manifest, configs, architecture identified
- [x] diffusers main + transformers 5.17 overlay installed and importing
- [x] fp8 quantiser written and validated at real shape on CPU
- [x] hipBLASLt root cause found; env var fix established
- [ ] **download complete** (was 8.7 / 33.1 GB when the GPU pool died)
- [ ] GPU restored by WSL restart
- [ ] fp8 tensors confirmed transferable to GPU (blocked on the broken pool)
- [ ] first real image generated

After a WSL restart the `.incomplete` shards in
`~/workspace/models/Qwen/Qwen-Image-2.1/.cache/huggingface/download/` resume automatically —
just re-run the downloader.

```bash
cd ~/workspace/qwen-image
source scripts/env.sh && qip_check                 # confirm GPU is healthy again
setsid nohup $QIP_VENV/bin/python -u scripts/download.py \
    > logs/download.log 2>&1 < /dev/null &         # resume download
~                                                     # then validate + generate
source scripts/env.sh && $QIP_VENV/bin/python scripts/test_fp8_real.py
source scripts/env.sh && $QIP_VENV/bin/python scripts/run_qwen_image.py \
    --prompt "..." --output out.png
```

**Never re-run a benchmark with hipBLASLt enabled** — it is what exhausted the GPU pool.

---

## 7. Update after the WSL restart: a WSL restart does *not* clear the GPU pool

`wsl --terminate archlinux` was executed (confirmed: boot uptime reset to 0). The download
shards survived and resumed fine. **But the GPU was still unusable afterwards** — an 8-byte
bf16 allocation still failed.

So the leaked state is **not** inside the WSL VM; it survives the VM being torn down and lives
on the **Windows-side graphics driver**. Diagnostics confirm it is not resource pressure:

* Windows reports `AMD Radeon RX 9070 XT` as `Status OK`; no stuck process (top user is `vmmem`
  at ~3.9 GB, then browsers).
* Host has 20.2 GB free of 31.1 GB, and no new GPADL failures are logged on the fresh boot at
  all (only benign `dxgkio_query_adapter_info: Ioctl failed: -22` lines).

Recovery therefore needs a **Windows-level graphics driver reset**, which requires
administrator rights. This WSL user is **not** an administrator
(`IsInRole(Administrator)` returns `False`), so it cannot be done from inside WSL.

### Fix (run on Windows, as Administrator)

Either, the light option — restart only the display adapter:

1. `Win+X` → **Device Manager** (or `devmgmt.msc`)
2. Expand **Display adapters** → right-click **AMD Radeon RX 9070 XT** → **Disable device**
3. Wait a few seconds → right-click → **Enable device**

(Disabling the adapter blanks the screen briefly and can move windows — expected.)

Or the reliable option: **reboot Windows.** Then re-enter WSL and check:

```bash
cd ~/workspace/qwen-image && source scripts/env.sh && qip_gpu_ok
```

`qip_gpu_ok` prints `GPU OK - allocations work` when the pool is usable. It probes in a
subprocess on purpose: a failed allocation poisons the in-process caching allocator.

### Do not repeat this

The trigger is any code path that hammers failed `hipModuleLoad` calls. Rule 1 in `AGENTS.md`
(`TORCH_BLAS_PREFER_HIPBLASLT=0`) exists to prevent it, and `env.sh` sets it. Never run a
benchmark with that variable unset.

---

## 8. A second environment bug: bracketed `[::1]` in `no_proxy`

This machine sets `no_proxy=localhost,127.0.0.1,::1,[::1]`. The httpx version in `./pylibs`
parses `no_proxy` entries as URLs and rejects the bracket, so **every** Hub call died with

```
httpx.InvalidURL: Invalid port: ':1]'
```

raised from deep inside `huggingface_hub` (`load_config`, `from_pretrained`, ...). `env.sh` now
strips `[`/`]` from `no_proxy`/`NO_PROXY` before anything imports httpx.

---

## 9. Validation that does not need the GPU

Because the GPU is down and the download is slow, the whole inference path was validated on
CPU instead:

* `scripts/test_fp8.py` — 26-Linear toy module: quantise, forward, `.to(dtype)` immunity,
  `nn.ModuleList` (`to_out`) handling.
* `scripts/test_fp8_real.py` — the **real** 7.115 B config: 232 Linears, 13.253 -> 6.632 GiB.
* `scripts/test_pipeline_cpu.py` — a miniature but structurally identical
  `QwenImage21Pipeline` (real classes, real tokenizer/processor, real scheduler) driven
  end-to-end: `encode_prompt` -> fp8 quantise -> denoise loop -> VAE decode -> PIL image,
  **once with `use_kv_cache=True` and once with `False`**. Both pass.

Along the way it pinned down the real contracts the runner now relies on:

* `encode_prompt` returns `(prompt_embeds, prompt_embeds_mask, image_pad_mask)` and **nulls the
  attention mask when it is entirely valid** (no padding), so `prompt_embeds_mask` is normally
  `None` for a single text prompt. Do not assume it is a tensor.
* `image_pad_mask` comes back as an all-`False` tensor for text-to-image (it marks real
  `<|image_pad|>` positions, of which a text-only prompt has none).
* The pipeline can be constructed directly (`QwenImage21Pipeline(scheduler=..., vae=...,
  text_encoder=..., processor=..., transformer=...)`) and accepts pre-computed `prompt_embeds`,
  which is what lets the runner free the text encoder before denoising.

`run_qwen_image.py` also gained `--device cpu`, so once the download finishes a real image can
be produced (slowly) even if the GPU is still down.

---

## 10. The runner is now validated end-to-end (still no GPU needed)

A tiny but complete on-disk model tree (`scripts/make_tiny_model.py`, 54 MiB, real
tokenizer, same class names and subfolder layout) lets `run_qwen_image.py` be exercised for
real. Three bugs were found and fixed this way, each of which would otherwise have surfaced
only after the 33 GB download:

1. **`--min-free-gib` was accepted but never used.** The VRAM guard was effectively
   hard-coded; the flag did nothing. It is now wired into `require_free(..., floor=True)`
   for the two highest-risk phases (text-encoder load and the denoise loop).

2. **Staged loading was wrong.** `QwenImage21Pipeline.from_pretrained(model_dir)` loads
   transformer (13.25 GiB) + text encoder (7.26 GiB) + VAE together into host RAM, about
   21.2 GiB, against ~21 GiB available. The text encoder is not needed until phase 3.
   Components are now loaded one at a time (transformer -> quantise -> text encoder ->
   free it -> VAE), so the peak is the quantised transformer plus a single component.

3. **A `save_pretrained`/reload round-trip silently destroyed the quantisation.**
   `Fp8Weight` keeps its bytes in **non-persistent** buffers, so `save_pretrained` wrote only
   the fp32/bf16-shaped placeholders and the reload reconstructed **empty `nn.Linear`
   layers**. The layer-count assertion caught it (`fp8 round-trip lost layers: 29 -> 0`).
   The round-trip was unnecessary — staged loading already bounds host RAM — so it is gone,
   and the reason is recorded in a comment so nobody re-adds it.

Also learned: `DiffusionPipeline.from_pretrained` cannot express "load only some
components". A missing `model_index.json` entry trips its *expected modules* check, and a
`None` entry crashes its own loader (`TypeError: 'NoneType' object is not subscriptable`).
The pipeline is therefore constructed directly from parsed configs, which also means the
tokenizer files never need copying.

Verified against the fixture, all with `--device cpu`:

| path | result |
|---|---|
| `--quantize fp8` | 29 layers quantised, full pipeline runs, 16x16 RGB PNG written |
| `--quantize none` | runs, PNG written |
| `--negative-prompt --guidance 2.0` | both embeds encoded, runs, PNG written |

The generated PNGs are real images (256 distinct colours), not blank output.

### What is still unverified

Everything GPU-specific: that `qip_gpu_ok` recovers, that the fp8 **uint8** weights
`.to("cuda")` cleanly (the one thing that has never once succeeded on this machine), and that
a real 1024x1024 image comes out. Those need the Windows driver reset in
`GPU_RECOVERY_NEEDED.md`.

---

## 11. Download throughput: raise `max_workers` (an earlier conclusion here was wrong)

`scripts/download.py` originally used `max_workers=3`, reasoning that this link is
flaky so fewer connections would be more robust. That was wrong and expensive: the
proxy here is **per-connection limited but not aggregate limited**, so throughput
scales with concurrency.

Measured aggregate throughput, each connection pulling a distinct byte range of the
same large file over a 10-12 s window:

| connections | aggregate |
|---|---|
| 1 | 0.71 MB/s |
| 3 | 2.28 MB/s |
| 6 | 3.89 MB/s |
| 10 | 5.70 MB/s |
| 16 | 7.44 MB/s |
| 24 | **9.34 MB/s** |
| 32 | 9.69 MB/s (saturated) |

The `max_workers=3` setting was delivering ~0.87 MB/s in practice, so roughly **10x of
the available bandwidth was being left unused** on a 33 GB download. `download.py` now
defaults to `max_workers=24` (override with `QIP_DL_WORKERS`), which moved the observed
rate to ~2.5 MB/s and the ETA from ~7.5 h to ~2.5 h.

An earlier measurement in this session concluded the opposite ("does not scale with
parallelism", "proxy is the limiter, not per-stream throttling"). That measurement was
invalid: it used `curl --max-time 12` without `-L`, so the 302 redirect to the CDN
consumed the whole window and every connection reported a few KB/s. Anyone re-measuring
this must follow redirects, use a time window long enough to get past connection setup,
and give each connection a distinct range so they do not contend for the same bytes.

(A synthetic 24-connection test reached 9.34 MB/s, but the real download settles nearer
2.5 MB/s. The difference is that the real run spreads its 24 workers across 7 different
files, and the six large shards dominate, so per-file concurrency is lower than the
synthetic test's 24-on-one-file. Raising the count further is not worth it past
saturation.)

---

## 12. CPU generation is a viable fallback (objective is not GPU-dependent)

Since the GPU has been down for the whole session, it is worth knowing whether the actual
objective — *produce a working generated image* — can be met without it. Measured on this
Ryzen 5 9600X (8 threads, bf16):

| shape | time | throughput |
|---|---|---|
| `1024x4096x12288` | 42.4 ms | 2431 GFLOP/s |
| `4096x4096x12288` | 168.4 ms | 2448 GFLOP/s |

That is ~2.4 TFLOP/s, only about 50x below the GPU's 128 TFLOP/s for the same op. A rough
forward-pass estimate for the 7.1B transformer at 1024x1024 (4096 tokens) is ~13 TFLOP, so
roughly **5 s/step, or ~3.5 min for 40 steps**, plus text encoding and decode.

Host RAM is the real constraint, but the fp8 quantisation is what makes it fit: the
transformer drops from 13.25 GiB to 6.63 GiB, so a CPU run can hold the quantised transformer
plus a working set inside the ~21 GiB available.

`run_qwen_image.py` already supports `--device cpu` and has been validated end-to-end against
the tiny fixture on that path, so once the download finishes a real image can be produced
without the GPU at all:

```bash
source scripts/env.sh && $QIP_VENV/bin/python scripts/run_qwen_image.py \
    --device cpu --height 1024 --width 1024 --steps 40 \
    --prompt "a red apple on a wooden table" --output cpu_out.png
```

This does not make the GPU path unnecessary — the fp8 **VRAM** win and the real throughput
still need the card — but it means the deliverable does not depend on the driver reset.

---

## 13. `fast_download.py`: multi-connection ranges cut the ETA from ~5 h to ~30 min

Raising `max_workers` on `snapshot_download` does **not** fix throughput, because
huggingface_hub uses **one HTTP connection per file** and merely spreads its pool across
files. With only 7 large files, the pool is idle: measured 1.3-1.9 MB/s while 24 workers
were "busy", yet 8 extra connections on the same link instantly added **4.87 MB/s**.

`scripts/fast_download.py` splits **each file** into 16 MiB ranges and fetches them over
many connections, which is what actually fills the pipe. Result on the 1.35 GB VAE file:
**6.24 MB/s average** (vs ~1.5 MB/s before), and ~11 MB/s on the large transformer shard —
about **6-7x**. Since files are processed largest-first and one at a time, all connections
concentrate on a single file, which is the peak-throughput case.

Design points that matter:

* **Resume is verified by content, not filename.** huggingface_hub names blobs
  `<hash>.<etag>.incomplete`, but the etag embedded there does **not** match the value in
  the sibling `.metadata` file, so blobs cannot be mapped to filenames by name. Two
  text_encoder shards even differ in size by only 32 bytes, so size matching is ambiguous
  too. Each candidate seed is therefore validated by comparing its first 1 MiB against a
  range request for the real file. All four in-flight prefixes proved distinguishable.
  Wrong seeds are rejected rather than silently producing a corrupt file.
* **sha256 comes from the Hub's LFS metadata**, and every completed file is verified before
  being moved to its final path. Independently confirmed: the completed `vae`, `tokenizer.json`
  and `qr.png` all match their published hashes.
* `urllib` failed through this machine's proxy with `SSLEOFError`; `requests` handles the
  CONNECT tunnel and redirects correctly, with a pooled `HTTPAdapter` so many ranges share
  keep-alive connections.
* `preflight.py` was taught to ignore `<name>.part` staging files, which share the
  `.safetensors` extension and would otherwise be counted as finished.

The old `snapshot_download`-based `download.py` is kept for reference but should not be
used for this link.

---

## 14. Small correctness/polish pass

* **VAE dtype is no longer hard-coded to fp32.** `run_qwen_image.py` loaded the VAE with
  `torch_dtype=torch.float32` for no established reason, which costs VRAM during decode while
  the fp8 transformer is still resident. There is a `--vae-dtype {auto,bfloat16,float16,float32}`
  option now, defaulting to `auto` (follows `--dtype`, i.e. bf16). Verified both paths against
  the tiny fixture. Note `_keep_in_fp32_modules` is `None` for both the VAE and the
  transformer, so no module requires fp32.
* **`preflight.py` no longer double-counts in-flight bytes.** It was adding the 9.9 GB of
  abandoned `.incomplete` blobs to the live `.part` progress. It now reports both mechanisms
  separately and takes the larger rather than the sum, since only one is active at a time and
  `.part` supersedes the blob for the same target.
* **The GPU allocation probe is now a single shared `gpu_alloc_works()` in `fp8_quant.py`.**
  It had been copy-pasted into three test files. Still subprocess-isolated on purpose: a failed
  HIP allocation poisons the calling interpreter's caching allocator.
* **Operational lesson recorded in `fast_download.py`:** redirect its output to a log file
  rather than piping it through `tail`. Piping buffers until exit, so a long download shows no
  progress at all; the `<name>.part` sizes in the model directory are the reliable progress
  signal.

All tests green after the pass: `test_fp8.py` (unit), `test_pipeline_cpu.py` (full tiny
pipeline, both KV-cache settings), `test_fp8_real.py` (real 7.115 B config).

---

## 15. Known cosmetic warning (not a bug)

`run_qwen_image.py` emits, at the start of the denoise phase:

```
FutureWarning: Accessing config attribute `text_encoder` directly via 'QwenImage21Pipeline'
object attribute is deprecated. Please access 'text_encoder' over 'QwenImage21Pipeline's
config object instead, e.g. 'scheduler.config.text_encoder'.
```

It comes from `DiffusionPipeline._execution_device` -> `self.components`, which does
`getattr(self, key)` for **every** key in `config`. `register_modules()` records
`config["text_encoder"] = (library, class)` but only creates the instance attribute when it
actually loads a module, so for components the runner attaches manually the lookup can fall
through to `ConfigMixin.__getattr__` and warn.

Deliberately **not** chased further: the attribute *is* set correctly by
`pipe.text_encoder = ...` (verified: `components['text_encoder'] is text_encoder` -> True, and
`__dict__` holds it), outputs are correct, and a reproduction in isolation raises no warning
at all. Attempts to silence it via `__dict__` assignment were reverted - that made the lookup
worse rather than better, and plain `setattr` is what `DiffusionPipeline.__setattr__` expects.

If it becomes noise, suppress narrowly rather than restructuring component loading.

---

## 16. Checkpoint facts confirmed from the downloaded shards

Read straight out of the two completed transformer shards:

* every tensor is **BF16** (211 + 86 = 297 tensors)
* the safetensors index resolves cleanly: 0 tensors missing, 0 extra, across both shards
* summing the tensor `data_offsets` spans gives **13.2530 GiB exactly**, and there is
  **zero aliasing** (no two tensors share bytes)
* `13.2530 GiB / 2 bytes = 7.1151 B` parameters, which matches
  `sum(p.numel() for p in model.parameters())` on a config-built model

So the transformer is **7.1151 B parameters / 13.253 GiB in bf16**, and the fp8 floor is
**6.632 GiB**. That is the number the VRAM budget is built on.

Correction to an earlier version of this section: it claimed the weight count was 6.958 B
and that the 7.115 B figure was inflated by non-persistent buffers. **That was wrong.** The
mistake was subtracting the safetensors header length from the file size and comparing that
to the parameter count - the header is real, and the correct comparison is against the sum
of the `data_offsets` spans, which gives 13.2530 GiB. There are no shared or tied tensors.

The method matters here: comparing file size to parameter count is unreliable because of the
header, and because shards may or may not alias. Summing `data_offsets` spans and checking
for overlaps is the reliable check.

`scripts/verify_checkpoint.py` now turns these into a hard gate: it loads both shards, applies
fp8, runs a real forward pass and checks the output is finite, loads the VAE, and optionally
does the same on GPU. It is the last check before spending time on a full run.

---

## 17. Progress accounting in `preflight.py`, take two

The previous fix (section 14) made the percentage *worse*: it added the live `.part` staging
to the leftover `.incomplete` blobs, which are not remaining work at all. That produced
"111.7% of total bytes fetched".

Correct model:

* `weights finished` = sum of final files that match their expected size
* `in-flight` = bytes in `.part` staging files (this is genuinely remaining work)
* `leftover .incomplete blobs` = seed material from `snapshot_download`; reported separately
  and explicitly marked as deletable to reclaim disk. Never added to progress.
* percentage = `(finished + in-flight) / total`, clamped to 100

Now reports a coherent `weights 26.15 / 33.12 GB (79.0%) finished`, `in-flight 0.34 GB`,
`-> 80.0% of total bytes fetched`.

Lesson worth keeping: when an accounting bug is "fixed", re-check that the new number is
*sane*, not just different. A >100% figure should have been caught immediately.

---

## 18. MILESTONE: first real generated image

`cpu_first.png` (256x256 RGBA) - "a red apple on a wooden table", produced by the real
33 GB checkpoint with fp8 transformer weights, running on CPU. 4 denoising steps,
36.5 s/step, 191 s total. 31,757 unique colours; mean RGB (177, 84, 65) matches the red
apple; strong row-to-row structure (6.45) confirms a real render rather than noise.

So the pipeline is proven end to end with real weights: text encoder -> fp8 transformer
denoise loop -> VAE decode -> PIL image.

### Runtime observed on CPU

| stage | time |
|---|---|
| transformer load (both shards, 13.25 GiB bf16) | 7 s |
| fp8 quantisation (232 layers) | ~25 s |
| text encoder load (16.33 GiB) + encode | 2.4 s + 15.7 s |
| VAE load | 1.0 s |
| denoise | 36.5 s/step at 256x256 |

Peak host RAM stayed comfortable: the text encoder (16.33 GiB) was freed before the VAE
loaded, exactly as the staged design intends.

### The transformer forward contract (hard-won, worth keeping)

Getting a standalone forward pass right took several attempts. The real contract, captured
from an instrumented pipeline call rather than inferred:

```
repeats = torch.where(img_mask, 4, 1)
joint   = cat([encoder_hidden_states, zeros(target_tokens // 4)], dim=1)
joint   = joint.repeat_interleave(repeats, dim=1)
joint[:, repeat_interleave(img_mask, repeats)] = hidden_states
```

Observed for a pure text-to-image sample (4x4 latents, 11 text tokens):

* `encoder_hidden_states` `(1, 11, C)`
* `img_mask` `(1, 15)` **bool**, with the **trailing** 4 entries `True`
* `hidden_states` `(1, 16, C)` == 4 x (number of `True`)
* `img_shapes` `[[(1, 4, 4)]]`

Invariants that must hold:

1. `len(img_mask) == text_tokens + target_tokens // 4`
2. `hidden_tokens == 4 * int(img_mask.sum())`  - the image tokens are placed into the
   expanded `True` positions, so that count must match exactly
3. `img_mask` must be **bool** (`torch.where` rejects Long), and
   `sum(prod(s) for s in img_shapes) == int(img_mask.sum())`

Wrong turns that cost time here, recorded so they are not repeated: assuming `img_mask`
was all-False for t2i (it is not - the target slots are `True`); assuming
`img_mask` width must differ from the joint length (they are equal); and reasoning from the
tiny fixture's VAE scale factor, which I had set to 4 while the real one is 16.
Monkey-patching the transformer's `forward` to capture its arguments settled it in one run.

---

## 19. `--fp8-cache-dequant`: 2x faster, but memory-hungry (measured)

Without a cache, every denoising step re-dequantises all 232 fp8 weights to bf16 - the same
work 40 times over. `Fp8Weight` can retain the materialised bf16 copy instead.

Measured on the real transformer, CPU, 256 latent tokens:

| | s/step | peak RSS |
|---|---|---|
| uncached | 36.5 | ~7 GiB |
| `--fp8-cache-dequant` | **18.4** | **22.0 GiB** |

So it is a genuine ~2x win, but the cache is a *second full bf16 copy* of the weights
(~13.25 GiB), which on this 23 GiB box leaves almost nothing. `run_qwen_image.py` therefore
**refuses** the flag unless there is real headroom (>= 20 GiB free host RAM, or >= 14 GiB free
VRAM), logs that it refused, and continues uncached. Verified: at 18.6 GiB free it declines
rather than risking an OOM.

If you want the 2x on CPU, free memory first; the guard is deliberately conservative because
an OOM on this machine has taken the host down before.

---

## 20. Second milestone image, and a correction on the cache's value

`cpu_cabin.png` (512x512 RGBA) - "a cozy cabin in a snowy pine forest at dusk, warm light in
the windows". 8 steps, 726 s denoise, 90.8 s/step, 793 s total. The prompt is followed
faithfully (cabin, pines, snow, dusk palette, warm lit windows), which is the strongest
evidence so far that the fp8 weights preserve model quality.

Note the run **refused** the dequant cache again (logged `REFUSING the cache`) because free
host RAM was below the 20 GiB threshold - the guard behaving as designed under load.

### Correction: the 2x figure only holds at small token counts

Section 19 reported the cache giving 36.5 -> 18.4 s/step. That was at **256** latent tokens.
At **512x512 (1024 tokens)** the uncached cost is 90.8 s/step, i.e. ~3.6x the 256-token cost
for 4x the tokens - so the uncached path scales close to linearly and the dequant is NOT the
dominant cost at larger sizes. The 2x was a small-token-count effect where per-call overhead
dominated; do not expect it at realistic resolutions.

Practical guidance: on CPU, budget roughly **90 s/step at 512x512** uncached. A 40-step
512x512 image is about an hour. The cache is only worth chasing when memory is plentiful and
tokens are few.

---

## 21. Real bug found: `--steps 1` silently produces a blank image (now guarded)

While testing whether the model handles its native 1024x1024 resolution on CPU, the run
completed "successfully" and wrote a **completely blank** PNG: 1 unique colour, std 0.0.
The run reported no error.

Root cause is in `FlowMatchEulerDiscreteScheduler`, not in this code:

* the scheduler config has `shift_terminal=0.02` and `use_dynamic_shifting=True`
* `_stretch_to_terminal` computes `scale_factor = one_minus_z[-1] / (1 - shift_terminal)`
* with `num_inference_steps=1` the sigma grid ends such that `scale_factor == 0`, so it
  divides by zero (emitting `RuntimeWarning: invalid value encountered in divide`)
* the resulting schedule is literally `sigmas = [nan, 0.0]`

NaN sigmas give NaN latents, which decode to a flat image. Verified directly:

| steps | sigmas | result |
|---|---|---|
| 1 | `[nan, 0.0]` | 1 unique colour, std 0.0 |
| 2..40 | finite, `scale_factor=1.02041` | normal image |

`run_qwen_image.py` now rejects `--steps < 2` with an explicit message instead of writing a
blank PNG. A blank output that reports success is the worst failure mode for an unattended
run, which is exactly the situation this deployment targets.

**Consequence for the 1024x1024 test:** it was NOT evidence about resolution. A correct
multi-step 1024x1024 run was still not attempted on CPU, because at 839 s/step (measured,
1 step) a 40-step run would take ~9 hours. That measurement is still useful: it confirms the
quadratic attention scaling (1024 tokens -> 90.8 s, 4096 tokens -> 839 s, i.e. 9.2x for 4x
the tokens) and it confirms 1024x1024 does not OOM on CPU.

Clean 12-step 256x256 control for comparison: 37,985 unique colours, 36.1 s/step.

---

## 22. Ready-to-run Windows GPU reset script

`Restart-GPU.ps1` disables and re-enables the AMD display adapter via `Disable-PnpDevice` /
`Enable-PnpDevice` (the `CIM` `Win32_VideoController` `Disable()` method is often unsupported
on modern drivers). It verifies the resulting status and prints the WSL-side check command.
Syntax validated with the PowerShell parser.

Run it as Administrator:
```
powershell -ExecutionPolicy Bypass -File Restart-GPU.ps1
```

---

## 23. Final status

### Working today

The deployment is functional and produces good images. Verified repeatedly from the real
33 GB checkpoint with fp8 transformer weights:

| artefact | detail |
|---|---|
| `cpu_first.png` | 256x256, "a red apple on a wooden table", 4 steps |
| `cpu_cabin.png` | 512x512, "a cozy cabin in a snowy pine forest at dusk, warm light in the windows", 8 steps |
| `cpu_strawberries.png` | 256x256, "a bowl of ripe strawberries on a rustic wooden table, soft daylight", 12 steps |

Download 100% complete and hash-verified (33.12 GB). `verify_checkpoint.py` passes:
7.115 B weights load across both shards, 232 Linears quantise 13.253 -> 6.632 GiB, the
forward pass returns finite values, VAE loads.

### The one gap, and why it is blocked

Running on the **RX 9070 XT itself** has never been possible in this session. The GPU cannot
allocate 8 bytes, from the very first probe onwards:

```
GPU UNUSABLE - cannot allocate 8 bytes.
```

The cause is upstream of any code in this repo: hipBLASLt's gfx1201 Tensile kernels fail to
load under WSL2/librocDXG, and each failed `hipModuleLoad` leaks the dxg GPU host-memory pool.
Once exhausted, all allocation fails and the state persists.

Established facts about the blocker:

* it **survives a WSL restart** (`wsl --terminate` was executed and verified: boot uptime
  reset, GPU still dead) - so it lives on the Windows-side graphics driver
* it is **not** resource pressure: Windows reports the adapter `Status OK`, 20+ GiB host RAM
  free, and no process holds the GPU
* the leak has **stopped growing** (5 benign boot-time dxg messages total, no new
  `establish_gpadl failed` entries)
* this user is **not** a Windows administrator (`IsInRole(Administrator)` returns `False`),
  so `Disable-PnpDevice` / `Enable-PnpDevice` cannot be run from inside WSL
* Docker Desktop's WSL integration is not enabled for this distro, so it offers no
  alternative GPU path

Consequently the fp8 **uint8 -> cuda -> view** transfer, and any GPU generation, remain
unverified. That is precisely what `scripts/validate_gpu.py` exists to settle (9 probes,
cheapest first), and `Restart-GPU.ps1` exists to perform the fix.

### To resume

On Windows, as Administrator, either run:

```powershell
powershell -ExecutionPolicy Bypass -File Restart-GPU.ps1
```

or reboot. Then, in WSL:

```bash
cd ~/workspace/qwen-image
source scripts/env.sh && qip_gpu_ok          # expect: GPU OK - allocations work
$QIP_VENV/bin/python scripts/validate_gpu.py # all 9 probes
$QIP_VENV/bin/python scripts/run_qwen_image.py --height 512 --width 512 --steps 10 \
    --prompt "a red apple on a wooden table" --output gpu_smoke.png
```

Budgeted from measured numbers: the fp8 transformer is 6.632 GiB, plus a 2.0 GiB KV cache and
~1.7 GiB of activations/VAE, i.e. ~10.3 GiB against 14.36 GiB free - comfortable headroom.

---

# PART 2: the GPU actually works. Three real bugs were hiding behind a false diagnosis.

## 24. RETRACTION: there was never a leaked dxg pool. It was my own env var.

For ten rounds this session reported that the GPU was unusable because failed
`hipModuleLoad` calls had leaked the WSL2 dxg GPU host-memory pool, and that only a Windows
driver reset could fix it. **That diagnosis was wrong.**

The actual cause was one line I had added to `scripts/env.sh` myself:

```
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

It breaks allocation on this ROCm build. A/B, same interpreter and driver:

| environment | result |
|---|---|
| clean env | `OK bf16 alloc` |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | `hipErrorInvalidValue` |

An 8-byte tensor failing with `hipErrorInvalidValue` looks exactly like an exhausted GPU
memory pool, which is why it was misread. The `dxg` kernel messages counted throughout the
session (5, `dxgkio_query_adapter_info: Ioctl failed: -22`) were benign boot-time noise, not
evidence of a leak - they never grew, and they did not reset across a Windows reboot, which
should have been the clue that they were unrelated.

Cost of the error: the user was asked to reboot Windows, and did. That was unnecessary.
`env.sh` no longer sets this variable, and the diagnostic message in `qip_gpu_ok` now checks
for it first.

Lesson: when a failure looks like resource exhaustion, verify by *removing your own
configuration* before blaming the platform. A clean-environment control test would have
found this in round 3 instead of round 12.

## 25. Bug 1: a transposed-view linear weight is ~825x slower

With the GPU finally reachable, a real transformer forward at 4096 tokens took **250 s**.
The cause is a layout trap:

| call | time | throughput |
|---|---|---|
| `x @ W.t()` (W is `(out, in)`) | 1010 ms | **0.10 TFLOP/s** |
| `x @ Wt` (Wt contiguous `(in, out)`) | 1.22 ms | **84.3 TFLOP/s** |
| `F.linear(x, W)` | 1000 ms | 0.10 TFLOP/s |
| `nn.Linear(x)` | 992 ms | 0.10 TFLOP/s |

`F.linear` and `nn.Linear` both hit the slow path because they transpose internally.
`TORCH_BLAS_PREFER_HIPBLASLT=0` does **not** fix this - the degenerate path is in the
fallback, not only in hipBLASLt.

Fix: `Fp8Weight` now materialises the weight as a contiguous `(in, out)` tensor
(`dequantize_transposed`) and computes `x @ Wt`.

Real model result: **250 s -> 3.2 s per forward at 4096 tokens, a 77x speedup.**

## 26. Bug 2: SDPA silently picks the MATH kernel, which OOMs at 1024x1024

For one `(1, 32, 4096, 128)` causal bf16 attention:

| backend | peak allocation |
|---|---|
| FLASH_ATTENTION | **0.06 GiB** |
| EFFICIENT_ATTENTION | 0.12 GiB |
| MATH | **4.91 GiB** <- selected by default |

Leaving PyTorch to choose gave the MATH kernel, which materialises the full score matrix.
That was the 5.06 GiB allocation that OOMed. diffusers' `native` backend dispatches to
`F.scaled_dot_product_attention`; the *kernel* choice is PyTorch's, and it picks badly here.
Noted in `env.sh` for anyone re-testing.

## 27. Bug 3: the VAE decode is the peak allocation, not the denoise loop

Even with attention fixed, 1024x1024 OOMed - in `autoencoder_kl_qwenimage21._decode`, not in
the transformer. The VAE has `decoder_base_dim=144` with 2x upsampling stages and is already
resident, so its decode is the largest single allocation in the run.

Fix: `pipe.vae.enable_tiling()` (note: `QwenImage21Pipeline` does **not** forward
`enable_vae_tiling()`, so calling that raises `AttributeError`; call the VAE directly).
Auto-enabled for GPU runs at 768px or more.

## 28. The staged-loading fix for GPU VRAM pressure

On CPU, staging works because components can be freed between phases. On GPU the fp8
transformer is *resident* (6.63 GiB of 15.82), so the 16.33 GiB text encoder cannot coexist
with it - the first GPU attempt failed with `need ~17.33 GiB for the text encoder, but only
8.43 GiB is available`.

Fix: encode the prompt **on CPU even in a GPU run** (~16 s measured, against minutes of
denoising). This keeps peak VRAM at transformer + KV cache + VAE only, and avoids shuffling
the transformer off the GPU and back (2 x 27 s).

Two guards also had to be corrected - both were double-counting already-resident weights:

* the VAE move required `--min-free-gib` (9.0) *in addition* to its own 0.63 GiB, refusing a
  healthy budget of 7.76 GiB free. The floor now applies only to the first GPU allocation.
* the denoise guard required the transformer's full size as *free* memory. It now guards on
  what the loop still has to allocate (KV cache + activations).

## 29. Result: working GPU generation

| config | steps | denoise | per step | peak free VRAM |
|---|---|---|---|---|
| 512x512 | 15 | 61.5 s | 4.10 s | 6.90 GiB |
| **1024x1024** | **40** | **137.7 s** | **3.44 s** | 7.38 GiB |

`gpu_1024.png` renders the official demo prompt faithfully - the neon sign text
"QWEN IMAGE 2.1" is spelled correctly, with wet-pavement reflections. No OOM, no hang.

`validate_gpu.py`: **10/10 probes pass**, including the fp8 `uint8 -> cuda -> view` round trip
(bit-exact) and load + quantise + move of the real 6.63 GiB transformer.

For reference, CPU took 90.8 s/step at 512x512 and 839 s/step at 1024x1024, so the GPU is
roughly 20-240x faster depending on resolution.

---

## 30. Bug 4: the KV-cache decision double-counted resident weights too

Same class of mistake as the two guards corrected in section 28, found while writing the
startup guide. The auto decision was:

```python
headroom = free / GIB - summary["total_GiB"] - 1.5 - 0.7   # WRONG
```

`free` from `torch.cuda.mem_get_info()` **already excludes** the resident transformer and
VAE, so subtracting `summary["total_GiB"]` again cancels it out. With 7.76 GiB genuinely
free the computed headroom was **-1.07 GiB**, so `use_kv_cache` was silently **off on every
single run** - including runs where it would have fitted comfortably.

Fixed to `headroom = free / GIB - 2.2` (a margin for activations and fragmentation). Now:

```
kv cache needs ~2.00 GiB; headroom ~5.26 GiB -> on
```

This matters most for the guidance path, where `true_cfg > 1` doubles the cache to 4.00 GiB
at 1024x1024; that now fits instead of being declined.

### Verified figures after the fix

| config | steps | denoise | per step |
|---|---|---|---|
| 512x512 | 15 | 56.9 s | 3.80 s |
| 1024x1024 | 20 | 72.5 s | 3.62 s |
| 1024x1024 | 40 | 137.7 s | 3.44 s |

Outputs are real renders: `preview.png` has 92,523 unique colours with mean RGB
(190, 117, 84) matching the red-apple prompt.

### The recurring lesson

Three separate bugs in this file were all "a quantity that already excludes X was compared
against a budget that also subtracted X". When adding a memory guard, check what the number
you are comparing against actually means; `mem_get_info().free` is what is *left*, not what
is *total*.

---

## 31. CORRECTION: hipBLASLt was never the real bottleneck. The weight layout was.

This session repeatedly claimed that `TORCH_BLAS_PREFER_HIPBLASLT=0` was responsible for a
"23,000x" speedup and that hipBLASLt was the root cause of the fp16/bf16 trouble. Re-measured
properly, **that attribution was wrong**. Here is the actual A/B (clean env, same driver,
`F.linear` vs a contiguous `(in, out)` matmul):

| shape | flag | `F.linear(x, W)` | `x @ Wt` (contiguous) |
|---|---|---|---|
| 64x1024x4096 | off | 908 ms (0.00 TF) | **0.048 ms** (11.2 TF) |
| 64x1024x4096 | on | 1832 ms (0.00 TF) | **0.076 ms** (7.1 TF) |
| 1024x4096x12288 | off | 972 ms (0.11 TF) | **1.044 ms** (98.8 TF) |
| 1024x4096x12288 | on | 1836 ms (0.06 TF) | **1.037 ms** (99.4 TF) |
| 4096x4096x12288 | off | 986 ms (0.42 TF) | **3.36 ms** (122.6 TF) |
| 4096x4096x12288 | on | 1816 ms (0.23 TF) | **3.35 ms** (122.9 TF) |

What this actually shows:

1. **The dominant factor is the layout, by ~1000x.** A transposed-view weight runs at
   0.1-0.4 TFLOP/s; a contiguous one at 99-123 TFLOP/s. The flag does not rescue the
   transposed case (0.42 TF off vs 0.23 TF on - both catastrophic).
2. **With a contiguous weight the flag is essentially irrelevant**: 122.6 vs 122.9 TFLOP/s.
   So `TORCH_BLAS_PREFER_HIPBLASLT` is a **no-op for performance** on this setup; it only
   makes the slow transposed path somewhat less slow (about 2x).
3. The earlier "23,000x" figure compared hipBLASLt-enabled `F.linear` against
   hipBLASLt-disabled `F.linear`, and then credited the flag. In reality both cells are the
   same broken path; the 0.049 ms number that made it look like a fix was the *contiguous*
   case. The flag did not provide that speedup - the layout did.

### So: is hipBLASLt "fixed"?

**No - it is bypassed, and only partially.** Facts established:

* the gfx1201 Tensile kernels are present on disk (32 `HH`/`BB` `.co` files in this
  torch wheel's `hipblaslt/library/gfx1201/`), but `hipModuleLoad` fails for them under
  WSL2/librocDXG with `named symbol not found` / `no kernel image is available for execution
  on the device`, followed by `Clearing modules and retrying hipModuleLoad`;
* these failures still appear in the logs during ordinary runs, so hipBLASLt is still being
  probed and still failing - it was never repaired;
* `torch.backends.cuda` exposes no hipBLASLt toggle in this build, so the only lever is the
  environment variable, and it buys ~2x on a path we no longer use;
* what actually made the model fast was **not** touching hipBLASLt at all, but making every
  linear weight contiguous `(in, out)` and calling `x @ Wt` (NOTES.md section 25), plus
  pinning the flash attention kernel (section 26).

Keeping `TORCH_BLAS_PREFER_HIPBLASLT=0` in `env.sh` is still reasonable - it avoids the
failed-lookup retry churn and is harmless - but it should not be described as the fix.

---

## 32. Is the hipBLASLt problem ROCm's fault or this machine's? — investigation

Question asked directly. Evidence gathered, and the conclusion distinguishes what is proven
from what is not.

### What the kernel files actually are

| location | `.co` files | format |
|---|---|---|
| `_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/` (pip wheel) | 144 | **143 CCOB v3.1**, 1 ELF |
| `/opt/rocm/core/lib/hipblaslt/library/gfx1201/` (AUR `rocm-gfx120x-bin`) | 146 | same, **CCOB** |
| `_rocm_sdk_libraries/lib/rocblas/library/` | 55 | **all plain ELF** |

`CCOB` is Clang's Code Object Bundle magic. It is **not** an anomaly: AMD's own AUR binary
package ships the identical layout, so this is simply how hipBLASLt kernels are packaged in
ROCm 7.14. A CCOB v3.1 bundle is zstd-framed (frame at offset 32), so the absence of plain
architecture strings in it is expected and is **not** evidence of corruption.

The rocBLAS side, by contrast, is plain ELF and greppable: `gfx1201`,
`amdgcn-amd-amdhsa--gfx1201`. That side works, which is consistent with it being ELF.

### What the failure actually is

The loader **finds the right files** and tries the expected variants in order:

```
TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co
TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201-xnack+.co
TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201-xnack-.co
```

and reports `named symbol not found` / `no kernel image is available for execution on the
device`, then retries and falls back. So it is a **kernel lookup failure inside the library**,
not a missing file and not a packaging mistake.

Note the naming mismatch in the request: the selector asks for a kernel named
`Cijk_Alik_Bljk_BBS_...` (BBS = bf16 compute) while the file it opens is named
`TensileLibrary_BB_BB_...` (BB = bf16 with HPA). Upstream's own naming is inconsistent here.

### Is it a local environment problem?

Evidence against:

* both hipBLASLt copies (AMD's AUR binary and AMD's pip wheel) have the same layout;
* the failure is in the library's search/symbol resolution, not in missing or corrupt files;
* there is a matching upstream report for exactly this hardware: **ROCm issue #6203,
  "[Issue]: RX 9070 / gfx1201: massive prefill slowdown unless ROCBLAS_USE_HIPBLASLT=0 is
  set"** — same GPU, same remedy.

Evidence I do **not** have: I have not run a native-Linux comparison on this machine myself.
The user reports that half precision works there; I am relying on that report, not on my own
measurement, so the "native Linux is fine" half of the picture is unverified by me.

Conclusion: **the defect is upstream in ROCm 7.14's hipBLASLt for gfx1201, not a
misconfiguration of this machine.** But that does not matter much in practice, because of the
next point.

### Does bypassing it cost anything? No.

With contiguous `(in, out)` weights, the two paths are equivalent:

| shape | ROCBLAS_USE_HIPBLASLT=0 | default |
|---|---|---|
| 1024x4096x12288 | 1.026 ms (100.4 TF) | 1.053 ms (97.9 TF) |
| 4096x4096x12288 | 3.321 ms (124.2 TF) | 3.109 ms (132.6 TF) |
| 512x4096x4096 | 0.637 ms (27.0 TF) | 0.620 ms (27.7 TF) |

hipBLASLt provides **no speed advantage on this GPU for these shapes**, so disabling it
loses nothing. It does help on the degenerate transposed-weight path (1943 ms -> 992 ms), but
that path is ~1000x slower than using contiguous weights regardless, so the layout fix
dominates.

`env.sh` now also sets `ROCBLAS_USE_HIPBLASLT=0` (rocBLAS can itself dispatch to hipBLASLt,
and this is the switch upstream's issue recommends). Verified no regression: 512x512 x 12
steps ran at 4.11 s/step with the KV cache on.

---

## 33. Persistent server (`scripts/server.py`)

Iterating on prompts with the one-shot CLI pays 27 s of transformer load + ~32 s of fp8
quantisation on *every* image. `server.py` loads the transformer and VAE once, keeps them
resident on the GPU, and serves a Gradio UI (plus a Gradio HTTP API) on port 7860.

Measured, 512x512 / 7 steps:

| | one-shot CLI | server |
|---|---|---|
| startup | - | ~60 s (load + quantise + move) |
| per image | +27 s load, +32 s quantise | **0** |
| request latency | ~90 s | **30-38 s** |

Request breakdown from the log: text encoder load 4 s -> encode 11 s -> denoise 23 s.

Design notes:

* **The text encoder is deliberately NOT kept resident.** At 16.33 GiB it cannot coexist
  with the resident transformer in ~14 GiB of VRAM. It is loaded, used, freed per request;
  page cache makes the reload ~4 s, against 23 s of denoising, so the trade is clearly
  worth it. `TextEncoderPool` is a context manager so the free always happens.
* **Generations are serialised with a lock.** Two concurrent denoise loops would not fit.
* **Startup waits for VRAM instead of dying.** The first server launch crashed with
  `not enough VRAM: need ~7.13 GiB, have 6.33 GiB` purely because another process held the
  GPU at that instant. It now polls for up to `--wait-for-vram` (default 120 s) and only
  gives up on a sustained shortage - a long-lived service should not be that fragile.
* **`scripts/pipeline_common.py`** holds the shared loading rules (config-only pipeline
  construction, CPU quantisation, staged component loading) so the CLI and the server cannot
  drift apart. `pipeline_common.encode_prompt` also documents the two easy-to-get-wrong
  contracts: `prompt_embeds_mask` is `None` when entirely valid, and `image_pad_mask` is an
  all-False tensor (not `None`) for text-to-image.
* VAE tiling is toggled per request based on size, since it is only needed at >=768px.

### A testing note

My first HTTP test reported a suspicious "15.0 s" for both requests. That was the test
client's fault, not the server's: it did not parse Gradio's SSE stream properly, so it
returned the first `data:` chunk instead of waiting for `event: complete`. The server log
showed the generations had actually run fine. Worth remembering that a wrong-looking
measurement should be checked against the server's own log before it is believed.

