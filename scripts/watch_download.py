#!/usr/bin/env python3
"""Append download progress to a log so a long transfer can be watched without blocking."""

import os
import time
from datetime import datetime

ROOT = os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
TOTAL = 33.13e9
LOG = os.path.expanduser("~/workspace/qwen-image/logs/download_watch.log")


def size() -> int:
    n = 0
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n


prev = size()
start = time.time()
first = prev
with open(LOG, "a") as fh:
    fh.write(f"\n=== watch started {datetime.now():%F %T} at {prev/1e9:.2f} GB ===\n")
    while True:
        time.sleep(120)
        cur = size()
        dt = time.time() - start
        rate = (cur - first) / dt if dt else 0
        eta = (TOTAL - cur) / rate if rate > 0 else float("inf")
        line = (
            f"{datetime.now():%H:%M:%S}  {cur/1e9:6.2f}/33.13 GB "
            f"({cur/TOTAL*100:5.1f}%)  now {rate/1e6:5.2f} MB/s  "
            f"eta {eta/3600:.1f} h"
        )
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
        prev = cur
        if cur >= TOTAL * 0.999:
            fh.write("=== complete ===\n")
            break
