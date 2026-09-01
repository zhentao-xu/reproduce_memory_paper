#!/usr/bin/env bash
# Environment prep for the Memory-R1 pipeline. Idempotent — safe to re-run.
# Handles: Python venv, pip install, HF cache dirs, model downloads (with proxy retry + mirror
# fallback). Once this exits 0, scripts/run_h100_qwen3_4b_pipeline.sh can run offline.
#
# Usage:
#   bash scripts/prep_env.sh                    # full prep — venv + install + models
#   bash scripts/prep_env.sh --skip-install     # skip pip install (assume deps ready)
#   bash scripts/prep_env.sh --skip-models      # skip model download (assume already present)
#   bash scripts/prep_env.sh --models-only      # skip venv/install, only fetch models
#
# Env overrides:
#   HF_ENDPOINT=https://hf-mirror.com   # use HF mirror if primary blocked by corp proxy
#   HF_HUB_OFFLINE=1                     # force offline mode (skip all HF network access)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

SKIP_INSTALL=false
SKIP_MODELS=false
MODELS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=true ;;
    --skip-models)  SKIP_MODELS=true ;;
    --models-only)  MODELS_ONLY=true; SKIP_INSTALL=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- HF cache env vars

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"
mkdir -p "$HF_HOME"

echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
[ -n "${HF_ENDPOINT:-}" ] && echo "  HF_ENDPOINT=$HF_ENDPOINT (using mirror)"

# ---------------------------------------------------------------- [1/3] Python + deps

if ! $MODELS_ONLY; then
    echo
    echo "[1/3] Python venv + deps"

    # Detect Python runner: prefer uv, then existing .venv, then system python.
    if command -v uv >/dev/null 2>&1; then
        if [ ! -d .venv ]; then
            echo "  → creating .venv via uv"
            uv venv .venv
        fi
        PY_RUN="uv run --no-sync"
        PIP_INSTALL="uv pip install"
    else
        if [ ! -d .venv ]; then
            echo "  → creating .venv via python -m venv"
            python3 -m venv .venv
        fi
        # shellcheck disable=SC1091
        source .venv/bin/activate
        PY_RUN="python"
        PIP_INSTALL="pip install"
    fi

    if ! $SKIP_INSTALL; then
        if [ -f requirements.txt ]; then
            echo "  → $PIP_INSTALL -r requirements.txt"
            $PIP_INSTALL -r requirements.txt
        fi
        # Editable install of the package itself so `python -m memory_r1.*` works.
        echo "  → $PIP_INSTALL -e ."
        $PIP_INSTALL -e . || echo "  ⚠ editable install failed; you may need to fix pyproject.toml"
    else
        echo "  → --skip-install: assuming deps are already installed"
    fi
else
    # For --models-only we still need a Python runner to call snapshot_download.
    if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
        PY_RUN="uv run --no-sync"
    elif [ -d .venv ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        PY_RUN="python"
    else
        PY_RUN="python"
    fi
fi

echo "  PY_RUN=$PY_RUN"

# ---------------------------------------------------------------- [2/3] Datasets

echo
echo "[2/3] Datasets"

if [ -f data/raw/locomo/locomo10.json ]; then
    echo "✓ data/raw/locomo/locomo10.json present"
else
    echo "⚠ data/raw/locomo/locomo10.json missing"
    echo "   LoCoMo is checked into git; try 'git pull' or copy from an online box."
fi

# ---------------------------------------------------------------- [3/3] Models

if $SKIP_MODELS; then
    echo
    echo "[3/3] Models — skipped (--skip-models)"
    echo
    echo "✅ Environment prep done."
    exit 0
fi

echo
echo "[3/3] Models — checking + downloading if missing"

# Look for a local flat dir first, then HF cache format. Returns the resolved abs path (or "").
find_local_model() {
    local repo_id="$1"     # e.g. Qwen/Qwen3-4B-Instruct-2507
    local flat_name="$2"   # e.g. qwen3-4b
    if [ -f "models/$flat_name/config.json" ]; then
        (cd "models/$flat_name" && pwd)
        return
    fi
    local org="${repo_id%%/*}"
    local name="${repo_id#*/}"
    local cache_dir="models/models--${org}--${name}"
    if [ -d "$cache_dir/snapshots" ]; then
        local snap_hash
        snap_hash="$(cat "$cache_dir/refs/main" 2>/dev/null || ls -1 "$cache_dir/snapshots" 2>/dev/null | head -1)"
        if [ -n "$snap_hash" ] && [ -f "$cache_dir/snapshots/$snap_hash/config.json" ]; then
            (cd "$cache_dir/snapshots/$snap_hash" && pwd)
            return
        fi
    fi
    if [ -f "models/$(basename "$repo_id")/config.json" ]; then
        (cd "models/$(basename "$repo_id")" && pwd)
        return
    fi
    echo ""
}

# Try snapshot_download with retries and, if the primary HF endpoint fails with proxy errors,
# fall back to the HF mirror (hf-mirror.com — a public HF replica sometimes reachable when
# huggingface.co is blocked).
try_download() {
    local repo_id="$1"
    local flat_dir="$2"
    local endpoint="$3"   # empty string = use default (huggingface.co)

    HF_ENDPOINT="$endpoint" $PY_RUN - <<PYEOF
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download

repo_id = "$repo_id"
flat_dir = "$flat_dir"
last_err = None
for attempt in range(1, 4):
    try:
        p = snapshot_download(repo_id, local_dir=flat_dir)
        print(f"  ✓ downloaded {repo_id} → {p}")
        sys.exit(0)
    except Exception as e:
        last_err = e
        print(f"  attempt {attempt}/3 failed: {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(3 * attempt)
print(f"  ✗ giving up on {repo_id}: {last_err}", file=sys.stderr)
sys.exit(1)
PYEOF
}

ensure_model() {
    local repo_id="$1"
    local flat_name="$2"
    local flat_dir="models/$flat_name"

    local resolved
    resolved="$(find_local_model "$repo_id" "$flat_name")"
    if [ -n "$resolved" ]; then
        echo "  ✓ $repo_id already present → $resolved"
        return 0
    fi

    if [ "$HF_HUB_OFFLINE" = "1" ]; then
        echo "  ✗ $repo_id missing and HF_HUB_OFFLINE=1 — cannot download"
        return 1
    fi

    echo "  ↓ $repo_id not found — trying primary HF endpoint..."
    if try_download "$repo_id" "$flat_dir" ""; then
        resolved="$(find_local_model "$repo_id" "$flat_name")"
        echo "  ✓ $repo_id resolved → $resolved"
        return 0
    fi

    echo "  ↓ primary failed; retrying via HF mirror (hf-mirror.com)..."
    if try_download "$repo_id" "$flat_dir" "https://hf-mirror.com"; then
        resolved="$(find_local_model "$repo_id" "$flat_name")"
        echo "  ✓ $repo_id resolved via mirror → $resolved"
        return 0
    fi

    echo "  ✗ Could not download $repo_id from either endpoint."
    return 1
}

FAILED=()
ensure_model "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b" || FAILED+=("Qwen/Qwen3-4B-Instruct-2507")
ensure_model "intfloat/e5-small-v2"        "e5-small-v2" || FAILED+=("intfloat/e5-small-v2")

if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    echo "❌ The following models could not be downloaded:"
    for m in "${FAILED[@]}"; do echo "   - $m"; done
    echo
    echo "  Options:"
    echo "  1) Set HF_ENDPOINT to your corp mirror and re-run:"
    echo "       HF_ENDPOINT=https://<your-mirror> bash scripts/prep_env.sh --models-only"
    echo "  2) Download on an online box + rsync the flat dir over:"
    echo "       python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='models/qwen3-4b')\""
    echo "       rsync -av models/qwen3-4b/ user@h100:$(pwd)/models/qwen3-4b/"
    echo "  3) If a proxy is available, export http_proxy/https_proxy first, then re-run."
    exit 3
fi

echo
echo "✅ Environment prep done — models ready under $HF_HOME/"
