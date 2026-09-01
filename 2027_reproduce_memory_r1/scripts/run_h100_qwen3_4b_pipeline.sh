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

fail_missing ".venv" \
    "Run 'uv sync' on an online box, then rsync .venv/ to this host. Alternatively, if the
     H100 box has PyPI access (even briefly), run 'uv sync --frozen' now."

echo "  (offline mode — HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE)"

if [ "$EVAL_ONLY" = "true" ]; then
    echo "[eval-only] Skipping data prep + training. Jumping to evaluation."
else
    # ---------------------------------------------------------------------- 1. data prep

    echo "[1/6] Building Manager training tuples (Algorithm 1) with heuristic extractor..."
    if [ -f data/processed/manager_train.jsonl ] && [ -f data/processed/locomo_train.jsonl ]; then
        echo "  ✓ data/processed/manager_train.jsonl already present. Skipping."
    else
        uv run python scripts/prepare_manager_data.py \
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
        uv run python scripts/prepare_answer_data.py \
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
        uv run python scripts/train_answer_agent.py \
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
        uv run python scripts/train_memory_manager.py \
            configs/paper_grpo_manager_h100_qwen3_4b.yaml
    fi
fi

# ---------------------------------------------------------------------- 5. eval

echo "[5/6] Evaluating on LoCoMo test set (1307 QA)..."
uv run python scripts/evaluate.py configs/eval_h100_qwen3_4b_no_openai.yaml

echo
echo "🏁 Full pipeline complete."
echo "   Answer Agent checkpoint: outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
echo "   Memory Manager checkpoint: outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
echo "   Predictions:  outputs/predictions_qwen3_4b_no_openai/"
