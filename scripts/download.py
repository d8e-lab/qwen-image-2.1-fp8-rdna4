#!/usr/bin/env python3
"""Robust resumable download of Qwen/Qwen-Image-2.1 with retries.

Concurrency is deliberately high. This link goes through a local proxy that is
per-connection limited but not aggregate limited: measured throughput on this
machine scaled from 0.71 MB/s on one connection to ~9.7 MB/s at 32, saturating
around 24. The original max_workers=3 therefore wasted most of the available
bandwidth (2.1 MB/s), which mattered a lot on a 33 GB download.

Saturation measurements (each connection a distinct byte range, 10-12 s window):
    N=1   0.71 MB/s
    N=3   2.28 MB/s
    N=6   3.89 MB/s
    N=10  5.70 MB/s
    N=16  7.44 MB/s
    N=24  9.34 MB/s
    N=32  9.69 MB/s   <- past this there is nothing left to win

Interrupting this script is safe: huggingface_hub keeps partial data in
`.cache/huggingface/download/*.incomplete` and resumes from there, and already
completed files are skipped via their cached metadata.
"""

import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# hf_xet can be fragile on flaky links; disable and use plain HTTP range requests.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "Qwen/Qwen-Image-2.1"
DEST = os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
MAX_WORKERS = int(os.environ.get("QIP_DL_WORKERS", "24"))

attempt = 0
while True:
    attempt += 1
    try:
        print(f"[attempt {attempt}] {time.strftime('%F %T')} start "
              f"(max_workers={MAX_WORKERS})", flush=True)
        p = snapshot_download(
            repo_id=REPO,
            local_dir=DEST,
            max_workers=MAX_WORKERS,
            etag_timeout=30,
        )
        print("DONE ->", p, flush=True)
        break
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[attempt {attempt}] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        time.sleep(min(30, 5 * attempt))
