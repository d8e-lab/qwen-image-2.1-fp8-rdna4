#!/usr/bin/env bash
# Fetch Qwen/Qwen-Image-2.1-PE-T2I (the prompt rewriter) into the local model tree.
#
# Two passes, because the two halves of the repo want different downloaders:
#   1. fast_download.py over the four ~5 GB shards: this link's proxy is per-connection
#      limited but not aggregate limited, so many byte ranges are what fills the pipe
#      (~9.7 MB/s vs ~2 MB/s for one connection per file).
#   2. snapshot_download for everything else (config, tokenizer, system_prompt.txt...),
#      ignoring the shards pass 1 already verified.
#
# Safe to re-run: pass 1 resumes/verifies, pass 2 skips files already present.
#
#   bash scripts/download_pe.sh            # foreground
#   setsid nohup bash scripts/download_pe.sh > logs/pe_download.log 2>&1 < /dev/null &
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

REPO="${PE_REPO:-Qwen/Qwen-Image-2.1-PE-T2I}"
# Deliberately inside the repo (which .gitignore already covers) rather than the shared
# ~/workspace/models tree: only this directory is mounted read-write in this environment,
# and $HOME/workspace/models is EROFS from here. `du -sh models` is the local model store.
DEST="${PE_DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/Qwen/Qwen-Image-2.1-PE-T2I}"

echo "=== [1/2] large shards ($REPO) -> $DEST"
# 5 shards x 8 range streams: the proxy is per-connection limited (~0.3-0.9 MB/s) but not
# aggregate limited, and all files are fetched concurrently.
"$QIP_VENV/bin/python" -u scripts/fast_download.py \
    --repo "$REPO" --dest "$DEST" --min-size 0.001 --workers "${PE_WORKERS:-40}" || exit 1

echo "=== [2/2] small files"
QIP_DEST="$DEST" QIP_REPO="$REPO" "$QIP_VENV/bin/python" -u - <<'EOF' || exit 1
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download(
    os.environ["QIP_REPO"],
    local_dir=os.path.expanduser(os.environ["QIP_DEST"]),
    ignore_patterns=["*.safetensors"],
    max_workers=8,
    etag_timeout=30,
)
print("DONE ->", p)
EOF

du -sh "$DEST"
