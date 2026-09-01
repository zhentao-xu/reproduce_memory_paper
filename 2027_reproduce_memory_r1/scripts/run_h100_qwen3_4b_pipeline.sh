#!/usr/bin/env bash
# End-to-end Memory-R1 pipeline on a single H100 with Qwen3-4B-Instruct-2507 + e5-small-v2.
# Open-source-only variant + OFFLINE-friendly — assumes no internet access on the H100 box.
#
# Everything is checked before download/install:
#   - Datasets: only LoCoMo needed. Skipped if data/raw/locomo/locomo10.json exists.
#   - Models: Qwen3-4B, e5-small-v2 — assumed pre-cached under ./models/ (HF_HOME).
#   - Python packages: only `uv sync --frozen --offline` — no PyPI reach.
#
# BEFORE RUNNING THIS SCRIPT on the offline H100 box, make sure the following are staged in
# the repo directory:
#   ./data/raw/locomo/locomo10.json                       (2.7 MB, checked into git)
#   ./models/models--Qwen--Qwen3-4B-Instruct-2507/        (~8 GB, download beforehand)
#   ./models/models--intfloat--e5-small-v2/               (~150 MB, download beforehand)
#   ./.venv/                                              (from `uv sync` on an online box)
#
# Wall-clock estimates on 1× H100 (80 GB):
#   Data prep (heuristic extractor + e5-small-v2):      ~ 5 min
#   Stage A: Answer Agent GRPO (200 steps):              ~ 3-5 hours
#   Stage B: Memory Manager GRPO (200 steps):            ~ 5-7 hours
#   End-to-end eval on LoCoMo test:                      ~ 30-45 min
#
# Usage:
#   bash scripts/run_h100_qwen3_4b_pipeline.sh                 # full pipeline
#   bash scripts/run_h100_qwen3_4b_pipeline.sh --stage-a-only  # skip Manager training + eval
#   bash scripts/run_h100_qwen3_4b_pipeline.sh --eval-only     # only re-run evaluation

set -euo pipefail

# Always run from the repo root, regardless of where the user invoked the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

# ---------------------------------------------------------------------- environment

# Force HuggingFace into offline mode so any accidental URL lookup falls back to local cache
# instead of hanging on a socket timeout.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export HF_HOME="${HF_HOME:-./models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-./models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-./models}"
export PYTHONUNBUFFERED=1

# Optional wandb — set WANDB_MODE=offline for an air-gapped run.
# export WANDB_PROJECT="memory-r1-reproduce"
# export WANDB_MODE="offline"

STAGE_A_ONLY=false
EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --stage-a-only) STAGE_A_ONLY=true ;;
    --eval-only) EVAL_ONLY=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------- helper: check-or-fail

fail_missing() {
    local path="$1"
    local hint="$2"
    if [ ! -e "$path" ]; then
        echo
        echo "❌ Missing prerequisite: $path"
        echo "   $hint"
        echo
        exit 2
    fi
    echo "✓ $path present"
}

# ---------------------------------------------------------------------- 0. sanity checks

echo "[0/6] Offline sanity checks..."

fail_missing "data/raw/locomo/locomo10.json" \
    "LoCoMo is checked into git; if you're missing it, git-pull or copy from an online box."

fail_missing "models/models--Qwen--Qwen3-4B-Instruct-2507" \
    "Download Qwen3-4B on an online box:
       HF_HOME=./models huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
     then scp/rsync ./models/models--Qwen--Qwen3-4B-Instruct-2507 to this host."

fail_missing "models/models--intfloat--e5-small-v2" \
    "Download e5-small-v2 on an online box:
       HF_HOME=./models huggingface-cli download intfloat/e5-small-v2
     then scp/rsync ./models/models--intfloat--e5-small-v2 to this host."

# Python runner: prefer uv if installed, otherwise fall back to plain python. This lets the
# script work both on a uv-managed box and on a corp box where uv isn't available and packages
# are pip-installed to --user / a plain venv.
if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
    PY_RUN="uv run --no-sync"
    echo "✓ uv + .venv detected — using 'uv run'"
elif [ -d .venv ]; then
    # Plain venv, activate it and use python directly.
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY_RUN="python"
    echo "✓ .venv detected — using activated python"
else
    # No venv; assume packages are installed globally (or via pip install --user).
    PY_RUN="python"
    echo "✓ using system python"
fi

# Sanity-check that the top-level deps are actually importable — otherwise the training will
# fail 5 min in with a cryptic ModuleNotFoundError.
if ! $PY_RUN -c "import torch, transformers, peft, trl, sentence_transformers, loguru" 2>/dev/null; then
    echo
    echo "❌ Python deps missing. Install them first:"
    echo "     pip install -r requirements.txt"
    echo "   Or for an isolated venv:"
    echo "     python3.12 -m venv .venv && source .venv/bin/activate"
    echo "     pip install -r requirements.txt"
    exit 3
fi
echo "✓ Python deps importable"

# Verify the ``memory_r1`` package is importable (installed via ``pip install -e .``). Without
# it, the entry scripts fail with 'ModuleNotFoundError: No module named memory_r1'.
if ! $PY_RUN -c "import memory_r1" 2>/dev/null; then
    echo "  ↳ memory_r1 package not installed; running 'pip install -e .' now..."
    $PY_RUN -m pip install -e . 2>&1 | tail -3
    if ! $PY_RUN -c "import memory_r1" 2>/dev/null; then
        echo "❌ 'pip install -e .' didn't make memory_r1 importable. Debug with:"
        echo "     $PY_RUN -c 'import memory_r1; print(memory_r1.__file__)'"
        exit 4
    fi
fi
echo "✓ memory_r1 package importable"

echo "  (offline mode — HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE)"

if [ "$EVAL_ONLY" = "true" ]; then
    echo "[eval-only] Skipping data prep + training. Jumping to evaluation."
else
    # ---------------------------------------------------------------------- 1. data prep

    echo "[1/6] Building Manager training tuples (Algorithm 1) with heuristic extractor..."
    if [ -f data/processed/manager_train.jsonl ] && [ -f data/processed/locomo_train.jsonl ]; then
        echo "  ✓ data/processed/manager_train.jsonl already present. Skipping."
    else
        $PY_RUN scripts/prepare_manager_data.py \
            --locomo data/raw/locomo/locomo10.json \
            --out data/processed/manager_train.jsonl \
            --extractor heuristic \
            --write-splits
    fi

    echo "[2/6] Building Answer Agent training tuples (Algorithm 2) with e5-small-v2..."
    if [ -f data/processed/answer_train.jsonl ]; then
        echo "  ✓ data/processed/answer_train.jsonl already present. Skipping."
    else
        # Uses ``--extractor heuristic`` (sentence splitter, no LLM) + intfloat/e5-small-v2 +
        # top-30 per speaker (60 total memories) — paper's retrieval budget.
        $PY_RUN scripts/prepare_answer_data.py \
            --locomo data/raw/locomo/locomo10.json \
            --out data/processed/answer_train.jsonl \
            --extractor heuristic \
            --encoder intfloat/e5-small-v2 \
            --top-k-per-speaker 30
    fi

    # ---------------------------------------------------------------------- 3. Stage A

    echo "[3/6] Stage A: RL fine-tuning the Answer Agent (200 steps, batch=128, G=8)..."
    STAGE_A_CKPT="outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
    if [ -d "$STAGE_A_CKPT" ]; then
        echo "  ✓ Stage A checkpoint already at $STAGE_A_CKPT. Skipping training."
    else
        $PY_RUN scripts/train_answer_agent.py \
            configs/paper_grpo_answer_h100_qwen3_4b.yaml
    fi

    if [ "$STAGE_A_ONLY" = "true" ]; then
        echo "[stage-a-only] Stopping after Stage A."
        exit 0
    fi

    # ---------------------------------------------------------------------- 4. Stage B

    echo "[4/6] Stage B: RL fine-tuning the Memory Manager (200 steps)..."
    STAGE_B_CKPT="outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
    if [ -d "$STAGE_B_CKPT" ]; then
        echo "  ✓ Stage B checkpoint already at $STAGE_B_CKPT. Skipping training."
    else
        # The manager config's answer_backend.checkpoint already points at Stage A's step_200.
        $PY_RUN scripts/train_memory_manager.py \
            configs/paper_grpo_manager_h100_qwen3_4b.yaml
    fi
fi

# ---------------------------------------------------------------------- 5. eval

echo "[5/6] Evaluating on LoCoMo test set (1307 QA)..."
$PY_RUN scripts/evaluate.py configs/eval_h100_qwen3_4b_no_openai.yaml

echo
echo "🏁 Full pipeline complete."
echo "   Answer Agent checkpoint: outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
echo "   Memory Manager checkpoint: outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
echo "   Predictions:  outputs/predictions_qwen3_4b_no_openai/"
