#!/usr/bin/env bash
# Environment prep for the Memory-R1 pipeline on an OFFLINE box (H100 without internet).
# Idempotent: safe to re-run. Skips work that's already done.
#
# Adapts to the runtime it finds:
#   1. If .venv exists → use it
#   2. Elif `uv` is available → create + use .venv via uv
#   3. Else → use system python; install deps via `pip install --user` (no venv)
#
# Validates (fails fast if anything is missing):
#   - Python deps importable (attempts pip install ONLY if missing + PIP mirror reachable)
#   - memory_r1 package importable
#   - LoCoMo dataset present at data/raw/locomo/locomo10.json
#   - Models present at models/qwen3-4b/ + models/e5-small-v2/
#
# Usage:
#   bash scripts/prep_env.sh              # full validation
#   bash scripts/prep_env.sh --no-install # never attempt pip install even if deps missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

NO_INSTALL=false
for arg in "$@"; do
  case "$arg" in
    --no-install) NO_INSTALL=true ;;
    --skip-install) NO_INSTALL=true ;;   # backward compat
    --skip-models) ;;                    # backward compat noop
    --models-only) ;;                    # backward compat noop
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- HF cache env vars
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"
mkdir -p "$HF_HOME"

echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE (offline mode)"

# ---------------------------------------------------------------- [1/4] Python runner

echo
echo "[1/4] Python runner"

if [ -d .venv ] && [ -x .venv/bin/python ]; then
    # Existing venv — activate + use its pip/python.
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="python"
    PIP="pip install"
    echo "  ✓ using existing .venv ($(python --version))"
elif command -v uv >/dev/null 2>&1; then
    echo "  → uv detected, no .venv — creating one via 'uv venv .venv'"
    uv venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="python"
    PIP="uv pip install"
    echo "  ✓ .venv created + activated ($(python --version))"
else
    # No uv, no venv — use whatever python3 the system provides. pip install --user for isolation.
    PY="python3"
    PIP="pip install --user"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "  ✗ no python3 in PATH — cannot proceed"
        exit 3
    fi
    if ! command -v pip >/dev/null 2>&1 && ! $PY -m pip --version >/dev/null 2>&1; then
        echo "  ✗ no pip / python -m pip — cannot install deps"
        exit 3
    fi
    # Prefer `python3 -m pip install --user` over bare `pip` (more consistent across systems).
    PIP="$PY -m pip install --user"
    echo "  ✓ using system python ($($PY --version)) + '$PIP'"
fi

export PY PIP  # so subshells can see them

# ---------------------------------------------------------------- [2/4] Python deps

echo
echo "[2/4] Python deps"

REQUIRED_MODULES=(torch transformers peft trl accelerate sentence_transformers loguru huggingface_hub datasets yaml)
MISSING_MODULES=()
for mod in "${REQUIRED_MODULES[@]}"; do
    if ! $PY -c "import $mod" 2>/dev/null; then
        MISSING_MODULES+=("$mod")
    fi
done

if [ ${#MISSING_MODULES[@]} -eq 0 ]; then
    echo "  ✓ all required modules importable (torch, transformers, peft, trl, ...)"
else
    echo "  ⚠ missing: ${MISSING_MODULES[*]}"
    if $NO_INSTALL; then
        echo "  ✗ --no-install set — cannot proceed with missing deps."
        exit 3
    fi
    if [ ! -f requirements.txt ]; then
        echo "  ✗ no requirements.txt found and deps are missing"
        exit 3
    fi
    echo "  → attempting: $PIP -r requirements.txt"
    if ! $PIP -r requirements.txt; then
        echo "  ✗ pip install failed — is the PIP mirror reachable on this box?"
        echo "    If truly offline, pre-install deps on an online mirror-attached machine first."
        exit 3
    fi
    # Re-check.
    STILL_MISSING=()
    for mod in "${MISSING_MODULES[@]}"; do
        $PY -c "import $mod" 2>/dev/null || STILL_MISSING+=("$mod")
    done
    if [ ${#STILL_MISSING[@]} -gt 0 ]; then
        echo "  ✗ still missing after install: ${STILL_MISSING[*]}"
        exit 3
    fi
    echo "  ✓ all deps installed"
fi

# ---------------------------------------------------------------- [2b/4] memory_r1 package

if $PY -c "import memory_r1" 2>/dev/null; then
    echo "  ✓ memory_r1 package importable"
else
    echo "  ⚠ memory_r1 not importable"
    if $NO_INSTALL; then
        echo "  ✗ --no-install set — cannot install memory_r1."
        exit 3
    fi
    echo "  → $PIP -e ."
    if ! $PIP -e .; then
        echo "  ✗ pip install -e . failed"
        exit 3
    fi
    if ! $PY -c "import memory_r1" 2>/dev/null; then
        echo "  ✗ memory_r1 still not importable after install (check pyproject.toml)"
        exit 3
    fi
    echo "  ✓ memory_r1 installed + importable"
fi

# ---------------------------------------------------------------- [3/4] Dataset

echo
echo "[3/4] LoCoMo dataset"
if [ -f data/raw/locomo/locomo10.json ]; then
    SIZE=$(du -h data/raw/locomo/locomo10.json | cut -f1)
    echo "  ✓ data/raw/locomo/locomo10.json ($SIZE)"
else
    echo "  ✗ data/raw/locomo/locomo10.json missing — try 'git pull'"
    exit 3
fi

# ---------------------------------------------------------------- [4/4] Models

echo
echo "[4/4] Models"

check_flat_model() {
    local name="$1"
    local flat_dir="models/$name"
    if [ -f "$flat_dir/config.json" ]; then
        local size
        size=$(du -sh "$flat_dir" | cut -f1)
        echo "  ✓ $flat_dir ($size)"
        return 0
    fi
    return 1
}

MISSING_MODELS=()
check_flat_model "qwen3-4b"    || MISSING_MODELS+=("qwen3-4b (Qwen/Qwen3-4B-Instruct-2507)")
check_flat_model "e5-small-v2" || MISSING_MODELS+=("e5-small-v2 (intfloat/e5-small-v2)")

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo
    echo "  ✗ Missing model dir(s):"
    for m in "${MISSING_MODELS[@]}"; do echo "     - models/$m"; done
    echo
    echo "    This box is offline. On an online machine (Mac / jump host), run:"
    echo "      bash scripts/download_models.sh"
    echo "    Then rsync the models dir over to this host:"
    echo "      rsync -av --info=progress2 models/ user@$(hostname):$(pwd)/models/"
    exit 3
fi

echo
echo "✅ Environment prep passed."
echo "   Python:   $($PY -c 'import sys; print(sys.version.split()[0])')"
echo "   PyTorch:  $($PY -c 'import torch; print(torch.__version__)')"
echo "   CUDA:     $($PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), "device(s)")')"
