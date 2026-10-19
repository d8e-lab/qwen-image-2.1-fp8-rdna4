# GPU troubleshooting (corrected)

> **History:** this file used to say the GPU was bricked by a leaked dxg host-memory pool that
> only a Windows driver reset could clear. **That was wrong.** The real cause was
> `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, which an earlier revision of
> `scripts/env.sh` set itself. That variable breaks allocation on this ROCm build, and an
> 8-byte tensor failing with `hipErrorInvalidValue` looks exactly like an exhausted pool.
> The variable is gone from `env.sh`. See NOTES.md section 24.

## If the GPU reports UNUSABLE, check in this order

### 0. Is `/dev/dxg` even visible? (a sandbox can hide it)

```bash
ls -l /dev/dxg            # must exist, major:minor 10:258 on this box
ls /dev | wc -l           # 6-8 entries = synthetic /dev, i.e. a sandbox
```

A restricted process sandbox mounts a **minimal synthetic `/dev`** (only `null`, `zero`,
`random`, `urandom`, `tty`, `full`, `pts`, `shm`). Under it `/dev/dxg` and `/dev/dri` do not
exist and `torch.cuda.init()` fails with `RuntimeError: No CUDA GPUs are available` — which
looks exactly like a dead GPU. It is not: the host's `dxgkrnl` is loaded and
`/sys/class/misc/dxg` (visible, because `/sys` comes from the host) still shows `10:258`.
Measured here: with the sandbox's synthetic `/dev` the probe fails, and from a process with
the real `/dev` the very same command prints `GPU OK - allocations work`.

So before touching drivers or rebooting anything, compare `/dev/dxg` with `/sys/class/misc/dxg`.
If the node is missing but the sysfs entry is present, the GPU is fine and the caller needs
a less restricted sandbox — not a Windows reset.

### 1. Your own environment (most likely)

```bash
cd ~/workspace/qwen-image
env | grep -i PYTORCH_HIP_ALLOC_CONF     # must print nothing
source scripts/env.sh && qip_gpu_ok
```

If `PYTORCH_HIP_ALLOC_CONF` contains `expandable_segments`, unset it and retry:

```bash
unset PYTORCH_HIP_ALLOC_CONF && qip_gpu_ok
```

Confirm with a clean-environment control before blaming anything else:

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin ~/workspace/torch_test/.venv/bin/python -c \
  "import torch;torch.cuda.init();t=torch.zeros(8,dtype=torch.bfloat16,device='cuda');\
   torch.cuda.synchronize();print('alloc OK')"
```

### 2. Only if the clean-environment test also fails: reset the adapter

As Administrator on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File Restart-GPU.ps1
```

or reboot. Then re-run step 1.

## Also worth knowing

* `dxgkio_query_adapter_info: Ioctl failed: -22` lines in `dmesg` on a fresh boot are
  **benign**. They do not grow, and they do not reset across a Windows reboot, so they are
  not a signal of anything wrong.
* A WSL restart (`wsl --terminate`) does not help if the cause is an environment variable,
  because the variable comes back from your shell profile or `env.sh`.
