#!/usr/bin/env python3
"""Parallel multi-connection downloader for Qwen-Image-2.1.

Why this exists
---------------
`huggingface_hub.snapshot_download` uses **one HTTP connection per file** and spreads
its worker pool across files. With only 7 large files, raising `max_workers` therefore
buys nothing: this link's proxy is per-connection limited (~0.3-0.7 MB/s each) but not
aggregate limited (24-32 connections reach ~9.7 MB/s total). Measured while the stock
downloader was running flat out at ~1.9 MB/s, 8 extra connections pulled another
4.87 MB/s, so most of the link was idle.

This script instead splits **each file** into many byte ranges and fetches them
concurrently, which is what actually fills the pipe. It resumes from the `.incomplete`
prefix that huggingface_hub already wrote, so no existing progress is lost, and it
verifies every finished file against the sha256 published in the Hub's LFS metadata
before writing it to its final path.

Usage:
    python scripts/fast_download.py            # all missing large files
    python scripts/fast_download.py --only <substr>

Operational note: redirect to a log file (``> logs/fast_download.log 2>&1``) rather than
piping through ``tail``. Piping buffers everything until the process exits, so a
long-running download shows no progress at all - monitor the ``<name>.part`` sizes in the
model directory instead, or tail the log.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import requests
from requests.adapters import HTTPAdapter

REPO = "Qwen/Qwen-Image-2.1"
DEST = os.path.expanduser("~/workspace/models/Qwen/Qwen-Image-2.1")
CACHE = os.path.join(DEST, ".cache", "huggingface", "download")
API = f"https://huggingface.co/api/models/{REPO}?blobs=true"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/main"
CHUNK = 16 * 1024 * 1024  # 16 MiB per range request


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def build_session() -> requests.Session:
    """A shared Session with connection pooling and pool-level retries.

    urllib failed through this machine's proxy with SSLEOFError; requests/urllib3
    handle the CONNECT tunnel and redirects properly. One Session with a large pool
    is what lets many range requests share keep-alive connections.
    """
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "qwen-image-fetch/1.0"})
    return s


def list_blobs(session: requests.Session) -> list[dict]:
    r = session.get(API, timeout=60)
    r.raise_for_status()
    info = r.json()
    out = []
    for s in info.get("siblings", []):
        lfs = s.get("lfs") or {}
        out.append({
            "rfilename": s["rfilename"],
            "size": s.get("size") or lfs.get("size") or 0,
            "sha256": lfs.get("sha256"),
        })
    return out


# Server-side sha256 of the first PREFIX_PROBE bytes of each large file, used to
# confirm that a resume seed really belongs to the file we are downloading. Blob
# names under .cache/huggingface/download embed an etag that does not match the
# value in the sibling .metadata file, and two text_encoder shards differ in size
# by only 32 bytes, so filename/size matching is ambiguous. Comparing content is
# definitive and costs one small range request.
PREFIX_PROBE = 1 << 20


def cached_blobs(info: dict) -> list[str]:
    """Candidate .incomplete blobs for this file, largest first.

    Only used as a resume seed; the caller validates the content before trusting it.
    """
    rel = info["rfilename"]
    folder = os.path.join(CACHE, os.path.dirname(rel))
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        if not f.endswith(".incomplete"):
            continue
        p = os.path.join(folder, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if 0 < sz <= info["size"]:
            out.append((sz, p))
    out.sort(reverse=True)
    return [p for _, p in out]


def seed_is_valid(session, url: str, seed_path: str) -> bool:
    """True if the seed's leading bytes match the real file's leading bytes."""
    n = min(PREFIX_PROBE, os.path.getsize(seed_path))
    if n == 0:
        return False
    with open(seed_path, "rb") as f:
        local = f.read(n)
    try:
        remote = fetch_range(session, url, 0, n - 1)
    except Exception as e:
        log(f"  could not validate seed prefix ({type(e).__name__}), treating as unusable")
        return False
    if local != remote:
        log(f"  seed prefix mismatch (wrong shard or stale) - seeding rejected")
        return False
    return True


def fetch_range(session: requests.Session, url: str, start: int, end: int,
                tries: int = 6) -> bytes:
    """Fetch [start, end] inclusive. Retries hard because the link is flaky."""
    for a in range(tries):
        try:
            r = session.get(url, headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            }, timeout=(30, 180))
            if r.status_code not in (200, 206):
                raise IOError(f"HTTP {r.status_code}")
            data = r.content
            if not data:
                raise IOError("empty body")
            return data
        except Exception as e:
            if a == tries - 1:
                raise
            wait = min(20, 1.5 * (a + 1))
            log(f"    retry {a+1}/{tries-1} for {start}-{end}: "
                f"{type(e).__name__} {str(e)[:80]}; sleeping {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_file(info: dict, session: requests.Session, workers: int,
                  verify: bool = True) -> bool:
    rel = info["rfilename"]
    size = info["size"]
    final = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    url = f"{RESOLVE}/{rel}"

    if os.path.isfile(final) and os.path.getsize(final) == size:
        log(f"  [skip] {rel} already complete")
        return True

    # Resume from huggingface_hub's partial data, but only after proving the seed
    # belongs to this file - a wrong seed would verify-fail at the end and waste the
    # whole download, which is exactly what we are trying to avoid.
    part = final + ".part"
    os.makedirs(os.path.dirname(part), exist_ok=True)
    have_part = os.path.isfile(part) and os.path.getsize(part) > 0
    if have_part and seed_is_valid(session, url, part):
        log(f"  resuming own .part ({os.path.getsize(part)/1e9:.2f} GB)")
    else:
        seeded = False
        for cand in cached_blobs(info):
            if seed_is_valid(session, url, cand):
                seed_len = min(os.path.getsize(cand), size)
                log(f"  seeding from {os.path.basename(cand)[:20]}... "
                    f"({seed_len/1e9:.2f} GB of prior progress, prefix verified)")
                with open(cand, "rb") as src, open(part, "wb") as dst:
                    dst.write(src.read(seed_len))
                seeded = True
                break
        if not seeded:
            if have_part:
                log("  existing .part unusable - restarting this file")
            open(part, "wb").close()

    done = os.path.getsize(part) if os.path.isfile(part) else 0
    done = min(done, size)
    if done >= size:
        if verify and info.get("sha256"):
            if _verify(part, info["sha256"]):
                os.replace(part, final)
                log(f"  [done] {rel} (verified)")
                return True
            log(f"  [bad] {rel} failed verification, restarting")
            os.remove(part)
            done = 0
        else:
            os.replace(part, final)
            log(f"  [done] {rel}")
            return True

    ranges = [(s, min(s + CHUNK - 1, size - 1)) for s in range(done, size, CHUNK)]
    total = size - done
    log(f"  {rel}: {done/1e9:.2f}/{size/1e9:.2f} GB done, fetching {total/1e9:.2f} GB "
        f"in {len(ranges)} ranges over {workers} connections")

    lock = threading.Lock()
    got = 0
    t0 = time.time()
    last = [0.0]

    def one(rng: tuple[int, int]) -> None:
        nonlocal got
        s, e = rng
        data = fetch_range(session, url, s, e)
        with lock:
            with open(part, "r+b") as f:
                f.seek(s)
                f.write(data)
            got += len(data)
            now = time.time()
            if now - last[0] > 15:
                last[0] = now
                rate = got / (now - t0)
                log(f"    {rel}: {got/1e9:.2f}/{total/1e9:.2f} GB  {rate/1e6:.2f} MB/s")

    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, ranges))
    except Exception as e:
        log(f"  [fail] {rel}: {type(e).__name__}: {e}")
        return False

    actual = os.path.getsize(part)
    if actual != size:
        log(f"  [fail] {rel}: size {actual} != expected {size}")
        return False
    if verify and info.get("sha256"):
        log(f"  verifying {rel} ...")
        if not _verify(part, info["sha256"]):
            log(f"  [fail] {rel}: sha256 mismatch")
            return False
    os.replace(part, final)
    dt = time.time() - t0
    log(f"  [done] {rel} ({size/1e9:.2f} GB in {dt/60:.1f} min = {total/dt/1e6:.2f} MB/s avg)")
    return True


def _verify(path: str, expect: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest().lower() == expect.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24, help="concurrent range requests in total")
    ap.add_argument("--only", default=None, help="substring filter on filenames")
    ap.add_argument("--min-size", type=float, default=0.05, help="only files bigger than this (GB)")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    session = build_session()
    blobs = list_blobs(session)
    big = [b for b in blobs if b["size"] >= args.min_size * 1e9]
    if args.only:
        big = [b for b in big if args.only in b["rfilename"]]
    big.sort(key=lambda b: -b["size"])
    if not big:
        log("nothing to do")
        return 0

    remaining = sum(
        max(0, b["size"] - (os.path.getsize(os.path.join(DEST, b["rfilename"]))
                            if os.path.isfile(os.path.join(DEST, b["rfilename"])) else 0))
        for b in big
    )
    log(f"{len(big)} large files, {remaining/1e9:.2f} GB remaining, {args.workers} connections")
    for b in big:
        log(f"  - {b['rfilename']}  {b['size']/1e9:.3f} GB  sha256={'yes' if b.get('sha256') else 'no'}")

    # Share the connection budget across files so each gets several streams.
    per_file = max(4, args.workers // max(1, len(big)))
    log(f"using {per_file} connections per file\n")

    t0 = time.time()
    failed = []
    for b in big:
        if not download_file(b, session, workers=per_file, verify=not args.no_verify):
            failed.append(b["rfilename"])
    log(f"\ntotal {time.time()-t0:.1f}s")
    if failed:
        log("FAILED: " + ", ".join(failed))
        return 1
    log("all requested files complete and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
