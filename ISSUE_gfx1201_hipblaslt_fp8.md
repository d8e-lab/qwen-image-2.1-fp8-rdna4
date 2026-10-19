# [gfx1201][WSL2] hipBLASLt's CCOB Tensile libraries fail to load (`hipErrorNoBinaryForGpu`, rc=209); fp8 GEMM is unusable

Ready-to-paste issue body for `ROCm/TheRock`. Environment and evidence collected on the
target machine; every command below was run as written.

---

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4), 15.82 GiB visible VRAM |
| Host | Windows 10 + WSL2 (Ubuntu-like Arch userspace) |
| ROCm | 7.14.0 from the pip wheels (`rocm-sdk-libraries` / `rocm-sdk-core` / `rocm-sdk-device-gfx1201`, all 7.14.0) |
| PyTorch | 2.12.0+rocm7.14.0 (hip 7.14.60850) |
| Python | 3.14.6 |
| hipBLASLt | `1.4.1`-era library from the 7.14.0 wheel, `_rocm_sdk_libraries/lib/hipblaslt/library/gfx1201/` |
| Env | `ROCBLAS_USE_HIPBLASLT=0`, `TORCH_BLAS_PREFER_HIPBLASLT=0` (the workaround from #6203) |

## Symptom

Any fp8 matmul fails, for every shape and both fp8 dtypes:

```python
import torch
a = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
b = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
s = torch.tensor([1.0], device="cuda")
torch._scaled_mm(a, b.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
```

```
CUDA error: HIPBLAS_STATUS_INTERNAL_ERROR when calling hipblasLtMatmul
  with transpose_mat1 t transpose_mat2 n m 128 n 128 k 512 ...
```

Tried: M/N/K = 16/64/64, 64/256/512, 128/128/512, 256/256/1024, 1024/1024/1024,
4096/4096/4096, 256/4096/4096; `float8_e4m3fn` and `float8_e5m2`. All fail.

## What the loader reports

```
hipModuleLoad failed: .../hipblaslt/library/gfx1201/TensileLibrary_F8F8_BF8_HA_Bias_SAB_SCD_SAV_UA_Type_F8B_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co
 error: no kernel image is available for execution on the device
hipModuleLoad failed: ...-xnack-.co        error: file not found
hipModuleLoad failed: ...-xnack+.co        error: file not found
getKernel failed: Cijk_Alik_Bljk_F8BS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs_MT16x16x64_...
 error: named symbol not found
```

With `TENSILE_DB=0xffff` the predicate matching is visible and **succeeds**
(`Solution index selected: 124920`), i.e. this is not the PCI-chip-ID predicate gating added
in 7.13 - the solution is selected and then its library cannot be loaded.

## The libraries are packaged as CCOB, and the inner object is fine on paper

`TensileLibrary_*.co` in the hipBLASLt `gfx1201` directory are CCOB v3.1 containers:

```
CCOB (offset 0)
 -> zstd frame at offset 32
 -> __CLANG_OFFLOAD_BUNDLE__ with 2 entries:
      host-x86_64-unknown-linux-gnu-      (0 bytes)
      hipv4-amdgcn-amd-amdhsa--gfx1201    (66,076,736 bytes, ELF, ET_DYN, EM_AMDGPU 224)
```

So the target arch, the container format and the payload are all what one would expect.

## A/B: a locally built gfx1201 code object loads, the shipped one does not

Built with the clang-23 from the *same* 7.14.0 wheel:

```bash
clang -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O2 -c k.cl -o k.co
ld.lld -shared -o k.hsaco k.co
```

then, in a process with an initialized HIP context:

```python
hip.hipModuleLoad(byref(mod), b"/tmp/k.hsaco")        # rc=0    OK
hip.hipModuleLoad(byref(mod), b"/tmp/fp8_inner.co")   # rc=209  hipErrorNoBinaryForGpu
```

(`fp8_inner.co` = the `hipv4-...gfx1201` entry extracted from the CCOB bundle above.)

Two differences in the packaged payload, and two failed attempts to work around them:

1. **`EI_ABIVERSION` is 2** in the CCOB payload, while a locally built object from the same
   clang-23 reports **4**:

   ```
   shipped  ...co:  OS/ABI AMDGPU - HSA   ABI Version 2   Type DYN   Machine EM_AMDGPU   Flags 0x4e, gfx1201
   local k.hsaco :  OS/ABI AMDGPU - HSA   ABI Version 4   Type DYN   Machine EM_AMDGPU   Flags 0x4e, gfx1201
   ```

   Patching the byte to 3 and to 4 and reloading still returns rc=209, so this is not only a
   single byte - the payload looks like it is in an older code-object layout despite the
   bundle entry being named `hipv4-amdgcn-amd-amdhsa--gfx1201`.

2. **No `amdhsa.target` in `NT_AMDGPU_METADATA`** (all three CCOB variants checked -
   `Type_BB`, `Type_I8I8`, `Type_F8B` - lack it; the local build carries
   `amdhsa.target: amdgcn-amd-amdhsa--gfx1201`). Injecting that key into the msgpack metadata
   and reloading also still returns rc=209.

So neither difference alone makes the object loadable, but both point at the packaging step
that turns Tensile kernels into the CCOB `.co` files.

## This is not fp8-only: hipBLASLt's CCOB libraries do not load at all here

Forcing hipBLASLt for a plain bf16 matmul (`ROCBLAS_USE_HIPBLASLT=1`, `TENSILE_DB=0xffff`):

```
hipModuleLoad failed: ...TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_...gfx1201.co
loaded code object .../hipblaslt/library/gfx1201/Kernels.so-000-gfx1201.hsaco   <-- fallback
Solution index selected: 108106
```

The GEMM then completes through the fallback, but ~1000x slower than rocBLAS on the same
shape (that is the effect reported in #6203). bf16 does not fail outright because rocBLAS's
own libraries are **plain `.hsaco`** (`.../rocblas/library/`), which load fine; fp8 has no
rocBLAS path, so there is nothing to fall back to.

## Impact

* fp8 matmul (`torch._scaled_mm`) is entirely unavailable on this configuration, so fp8 LLM
  and DiT inference cannot use the GPU's fp8 units at all.
* Forcing hipBLASLt for bf16/bf16 makes things ~1000x worse instead of better.

## Question

Is CCOB loading known to be broken under WSL2 (librocDXG), or is the missing `amdhsa.target`
in the packaged objects the actual defect? If it is the latter, are there 7.14.x/nightly
artifacts with corrected metadata, and is there an env switch to make hipBLASLt use the plain
`.hsaco` kernel path instead of the CCOB per-solution libraries?

Related: #7861 ([gfx1201][Windows] FP8 forward fails, 7.14.1 vs 7.14.0), #5723
([Windows][gfx1201] hipblasLt/`_scaled_mm` crash), #6203 (prefill slowdown unless
`ROCBLAS_USE_HIPBLASLT=0`).
