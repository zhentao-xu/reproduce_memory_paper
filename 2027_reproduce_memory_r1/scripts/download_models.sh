#!/usr/bin/env bash
# Download Memory-R1 models to flat local dirs, ready to rsync to an offline machine.
# Run this on ANY box with HF Hub access (your Mac, a jump host, etc.), then rsync
# the ./models/ folder to the offline H100.
#
# Produces:
#   ./models/qwen3-4b/       — Qwen/Qwen3-4B-Instruct-2507 (~8 GB, flat files — no symlinks)
#   ./models/e5-small-v2/    — intfloat/e5-small-v2 (~150 MB, flat files)
#
# Both dirs are safe to `rsync -av` (or `scp -r`) to the remote box. No HF cache
# tree, no symlinks — just plain files that transformers/sentence-transformers
# will load via the directory path.
#
# Usage:
#   bash scripts/download_models.sh
#
# Then copy to the H100:
#   rsync -av --info=progress2 models/ user@h100:/path/to/repo/models/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

mkdir -p models

# Pick a python: prefer .venv, then uv, then system python. huggingface_hub must be importable.
if [ -d .venv ] && [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PY="uv run --no-sync python"
else
    PY="python3"
fi

echo "  using $PY"

if ! $PY -c "import huggingface_hub" 2>/dev/null; then
    echo "❌ huggingface_hub not importable in $PY."
    echo "   Run: pip install huggingface_hub  (or: uv pip install huggingface_hub)"
    exit 1
fi

download_one() {
    local repo_id="$1"
    local flat_dir="$2"
    if [ -f "$flat_dir/config.json" ]; then
        echo "  ✓ $repo_id already at $flat_dir — skipping"
        return
    fi
    echo "  ↓ downloading $repo_id → $flat_dir"
    $PY - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("$repo_id", local_dir="$flat_dir")
print(f"  ✓ done: {p}")
PYEOF
}

download_one "Qwen/Qwen3-4B-Instruct-2507" "models/qwen3-4b"
download_one "intfloat/e5-small-v2"        "models/e5-small-v2"

echo
echo "✅ Models ready under $REPO_ROOT/models/"
echo
echo "Next: copy to the offline machine, e.g."
echo "  rsync -av --info=progress2 models/ user@h100:/path/to/repo/models/"
