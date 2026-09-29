#!/usr/bin/env bash
# Resume the Qwen-Image-2.1 download, detached so it survives the parent shell.
# huggingface_hub resumes from the .incomplete shards, so this is safe to run
# repeatedly and safe to interrupt (including by a WSL restart).
set -euo pipefail

QIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${QIP_VENV:-$HOME/workspace/torch_test/.venv}"

mkdir -p "$QIP_ROOT/logs"

if pgrep -f "[d]ownload.py" > /dev/null; then
  echo "downloader already running (pid $(pgrep -f '[d]ownload.py' | tr '\n' ' '))"
else
  setsid nohup "$VENV/bin/python" -u "$QIP_ROOT/scripts/download.py" \
      > "$QIP_ROOT/logs/download.log" 2>&1 < /dev/null &
  disown || true
  sleep 2
  echo "downloader started (pid $(pgrep -f '[d]ownload.py' | tr '\n' ' '))"
fi

setsid nohup "$VENV/bin/python" -u "$QIP_ROOT/scripts/watch_download.py" \
    > "$QIP_ROOT/logs/watch.out" 2>&1 < /dev/null &
disown || true

echo "progress -> $QIP_ROOT/logs/watch.out"
sleep 5
tail -n 3 "$QIP_ROOT/logs/watch.out" 2>/dev/null || true
