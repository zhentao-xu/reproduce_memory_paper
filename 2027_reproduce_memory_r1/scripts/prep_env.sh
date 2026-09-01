#!/usr/bin/env bash
# =============================================================================
#  prep_env.sh — one-shot environment setup for the Memory-R1 reproduction.
# =============================================================================
#
#  WHAT IT DOES (in plain English):
#    1. Creates a Python virtual environment (`.venv/`) in this checkout so all
#       dependencies stay isolated from the rest of your system.
#    2. Installs the Python libraries the training/eval code needs (PyTorch,
#       HuggingFace transformers, PEFT, TRL, sentence-transformers, ...).
#    3. Installs this repo itself as a package (`memory_r1`) in editable mode
#       so `import memory_r1` works from any script.
#    4. Verifies the LoCoMo dataset (the paper's benchmark, ships with the repo)
#       is on disk.
#    5. Verifies the two HuggingFace models we need are on disk:
#         - Qwen/Qwen3-4B-Instruct-2507  (the LLM being fine-tuned)
#         - intfloat/e5-small-v2         (the retriever embedder)
#
#  WHEN TO RUN IT:
#    Once, right after cloning the repo. Also safe to re-run any time — it
#    skips work that's already done.
#
#  RUN THIS BEFORE:
#    scripts/run_h100_qwen3_4b_pipeline.sh  (the actual training + eval pipeline)
#
#  WHAT IT NEEDS:
#    - A Python 3.10-3.12 interpreter available as `python3`.
#    - Network access to a PyPI mirror (to install libraries the first time).
#    - The two model directories under `models/` (see step 5 above). On an
#      offline box, download them on an online machine with
#      `scripts/download_models.sh` and rsync them over.
#
#  HOW IT DECIDES WHICH PYTHON TO USE (in order of preference):
#    1. If `.venv/` already exists → activate it.
#    2. Else if `uv` is installed AND it can fetch a cpython from GitHub →
#       create `.venv/` via `uv venv` (fast path).
#    3. Else fall back to `python3 -m venv .venv` (only needs PyPI, works on
#       boxes where GitHub is blocked — this is the recommended path for you
#       if uv can't reach the internet).
#    4. Else as a last resort, install packages with `pip install --user` and
#       no venv (legacy behavior).
#
#  USAGE:
#    bash scripts/prep_env.sh              # normal — full setup + validation
#    bash scripts/prep_env.sh --no-install # only validate; don't run pip
#    bash scripts/prep_env.sh --no-uv      # skip uv entirely; go straight to
#                                          # `python3 -m venv` (also enabled by
#                                          # exporting PREP_ENV_NO_UV=1)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

NO_INSTALL=false
NO_UV="${PREP_ENV_NO_UV:-false}"
for arg in "$@"; do
  case "$arg" in
    --no-install) NO_INSTALL=true ;;
    --skip-install) NO_INSTALL=true ;;   # backward compat
    --skip-models) ;;                    # backward compat noop
    --models-only) ;;                    # backward compat noop
    --no-uv) NO_UV=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done
# Normalize NO_UV to "true"/"false"; accept 1/yes/on as truthy.
case "$NO_UV" in
  1|true|TRUE|True|yes|YES|on|ON) NO_UV=true ;;
  *) NO_UV=false ;;
esac

# ---------------------------------------------------------------- HF cache env vars
# Tell the HuggingFace libraries three things:
#   1. Don't try to reach the internet (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
#      This is a safety belt: even if network is available, we want models loaded
#      strictly from local disk so runs are reproducible.
#   2. Look for cached models under this repo's `models/` directory, not the
#      user's global `~/.cache/huggingface/` (keeps the checkout self-contained).
#   3. Create that models/ directory if it's missing.
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
# Make sure we have a working Python interpreter with pip. The order of
# preference (existing .venv → uv → python3 -m venv → system python) is
# explained in the header comment. After this section runs successfully, two
# shell variables are set for the rest of the script:
#   PY   = the python interpreter command    (e.g. "python" inside a venv)
#   PIP  = the pip install command            (e.g. "pip install", "uv pip install")
# Every later section uses these two variables instead of hardcoding tool names.

echo
echo "[1/4] Python runner"

if [ -d .venv ] && [ -x .venv/bin/python ]; then
    # Existing venv — activate + use its pip/python.
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="python"
    PIP="pip install"
    echo "  ✓ using existing .venv ($(python --version))"
else
    # Need to create .venv. Prefer uv (fast). Fall back to stdlib venv when uv can't reach
    # GitHub to download a cpython (typical on offline boxes that still have a PyPI mirror).
    VENV_CREATED=false
    UV_ERR_LOG="$(mktemp -t uv_venv.XXXXXX)"

    if $NO_UV; then
        echo "  → --no-uv (or PREP_ENV_NO_UV=1) set — skipping uv, will use 'python3 -m venv' directly"
    elif command -v uv >/dev/null 2>&1; then
        echo "  → uv detected — trying 'uv venv .venv'"
        if uv venv .venv 2>"$UV_ERR_LOG"; then
            PIP="uv pip install"
            VENV_CREATED=true
        else
            echo "  ⚠ uv venv failed (typically means GitHub is blocked so cpython can't be fetched):"
            sed 's/^/      /' "$UV_ERR_LOG" | head -5
            rm -rf .venv 2>/dev/null || true
        fi
    fi
    rm -f "$UV_ERR_LOG"

    if ! $VENV_CREATED && command -v python3 >/dev/null 2>&1; then
        PY_VENV_ERR_LOG="$(mktemp -t py_venv.XXXXXX)"
        echo "  → creating .venv via 'python3 -m venv' (system Python: $(python3 --version))"
        if python3 -m venv .venv 2>"$PY_VENV_ERR_LOG"; then
            PIP="python -m pip install"
            VENV_CREATED=true
        else
            echo "  ⚠ python3 -m venv failed:"
            sed 's/^/      /' "$PY_VENV_ERR_LOG" | head -5
            echo "      (on Debian/Ubuntu you may need: 'apt install python3-venv')"
            rm -rf .venv 2>/dev/null || true
        fi
        rm -f "$PY_VENV_ERR_LOG"
    fi

    if $VENV_CREATED; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        # Bump pip inside the fresh venv so it can resolve modern wheels (torch, etc.).
        python -m pip install --upgrade pip >/dev/null 2>&1 || true
        PY="python"
        echo "  ✓ .venv created + activated ($(python --version))"
    else
        # Last resort: system python + --user (no venv). Retains legacy behavior for boxes
        # where neither uv nor `python3 -m venv` can produce a working environment.
        if ! command -v python3 >/dev/null 2>&1; then
            echo "  ✗ no python3 in PATH — cannot proceed"
            exit 3
        fi
        PY="python3"
        if ! command -v pip >/dev/null 2>&1 && ! $PY -m pip --version >/dev/null 2>&1; then
            echo "  ✗ no pip / python -m pip — cannot install deps"
            exit 3
        fi
        # Prefer `python3 -m pip install --user` over bare `pip` (more consistent across systems).
        PIP="$PY -m pip install --user"
        echo "  ⚠ no working venv path — using system python ($($PY --version)) + '$PIP'"
    fi
fi

export PY PIP  # so subshells can see them

# ---------------------------------------------------------------- [2/4] Python deps
# Check that all the third-party libraries the training/eval scripts need are
# importable in this Python. If any are missing, run
#   pip install -r requirements.txt
# from a PyPI mirror. If the pip install fails (no mirror reachable), exit —
# there's nothing we can do offline other than tell the user to install these
# on an online box first.

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
# Install THIS repo as an editable Python package so `import memory_r1` works.
# Editable install (`pip install -e .`) means: changes to the source files in
# `src/memory_r1/` take effect immediately without re-installing. This is what
# makes `python scripts/train_answer_agent.py ...` work — the training script
# imports from the memory_r1 package.

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
# LoCoMo is the long-form conversation benchmark from the paper (10 dialogues,
# ~1300 evaluation questions). It ships committed inside the repo, so if the
# JSON file isn't present, the git checkout is broken — recommend `git pull`.

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
# The pipeline needs two HuggingFace model directories on local disk:
#   - models/qwen3-4b       — the 4B-parameter LLM being fine-tuned with GRPO
#   - models/e5-small-v2    — the embedding model for retrieval augmentation
# These are ~8 GB and ~130 MB respectively. Downloading them requires internet
# (they aren't checked into git). If missing, this script tells you exactly
# how to download them on an online machine and rsync them here.

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
