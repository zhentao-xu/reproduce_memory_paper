#!/usr/bin/env bash
# Environment prep for the Memory-R1 pipeline on an OFFLINE box (H100 without HF Hub access).
# Assumes ./models/qwen3-4b/ and ./models/e5-small-v2/ have been rsync'd over from an online box
# (see scripts/download_models.sh for how to produce them). Does NOT attempt any HF network I/O.
#
# Handles: Python venv, pip install, HF cache env vars, model presence checks.
#
# Usage:
#   bash scripts/prep_env.sh                    # full prep — venv + install + validate models
#   bash scripts/prep_env.sh --skip-install     # skip pip install (deps already installed)
#   bash scripts/prep_env.sh --skip-models      # skip model presence check
#   bash scripts/prep_env.sh --models-only      # only validate models are present

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
# Force offline mode. Models must be present locally under ./models/.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"
mkdir -p "$HF_HOME"

echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE (offline mode)"

# ---------------------------------------------------------------- [1/3] Python + deps

if ! $MODELS_ONLY; then
    echo
    echo "[1/3] Python venv + deps"

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
        echo "  → $PIP_INSTALL -e ."
        $PIP_INSTALL -e . || echo "  ⚠ editable install failed"
    else
        echo "  → --skip-install: assuming deps installed"
    fi
else
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
    echo "⚠ data/raw/locomo/locomo10.json missing — try 'git pull'"
fi

# ---------------------------------------------------------------- [3/3] Models

if $SKIP_MODELS; then
    echo
    echo "[3/3] Models — skipped (--skip-models)"
    exit 0
fi

echo
echo "[3/3] Models — validating presence"

check_flat_model() {
    local name="$1"
    local flat_dir="models/$name"
    if [ ! -f "$flat_dir/config.json" ]; then
        return 1
    fi
    echo "  ✓ $flat_dir (config.json present)"
}

MISSING=()
check_flat_model "qwen3-4b"    || MISSING+=("qwen3-4b (Qwen/Qwen3-4B-Instruct-2507)")
check_flat_model "e5-small-v2" || MISSING+=("e5-small-v2 (intfloat/e5-small-v2)")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo
    echo "❌ Missing model dir(s):"
    for m in "${MISSING[@]}"; do echo "   - models/$m"; done
    echo
    echo "  This box is offline. Prepare the models on an ONLINE box first:"
    echo "     bash scripts/download_models.sh"
    echo "     rsync -av --info=progress2 models/ user@$(hostname):$(pwd)/models/"
    exit 3
fi

echo
echo "✅ Environment prep done."
