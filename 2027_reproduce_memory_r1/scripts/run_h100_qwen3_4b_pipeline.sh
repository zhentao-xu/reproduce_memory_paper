#!/usr/bin/env bash
# End-to-end Memory-R1 pipeline on a single H100 with Qwen3-4B-Instruct-2507 + e5-small-v2.
# Open-source-only variant (no OpenAI). Environment prep + model download live in the sibling
# script scripts/prep_env.sh so this file focuses only on data prep + training + eval.
#
# Wall-clock estimates on 1× H100 (80 GB):
#   Data prep (heuristic extractor + e5-small-v2):      ~ 5 min
#   Stage A: Answer Agent GRPO (200 steps):              ~ 3-5 hours
#   Stage B: Memory Manager GRPO (200 steps):            ~ 5-7 hours
#   End-to-end eval on LoCoMo test:                      ~ 30-45 min
#
# Usage:
#   bash scripts/run_h100_qwen3_4b_pipeline.sh                  # env prep + full pipeline
#   bash scripts/run_h100_qwen3_4b_pipeline.sh --skip-prep      # skip prep_env.sh
#   bash scripts/run_h100_qwen3_4b_pipeline.sh --stage-a-only   # skip Manager training + eval
#   bash scripts/run_h100_qwen3_4b_pipeline.sh --eval-only      # only re-run evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

STAGE_A_ONLY=false
EVAL_ONLY=false
SKIP_PREP=false
for arg in "$@"; do
  case "$arg" in
    --stage-a-only) STAGE_A_ONLY=true ;;
    --eval-only)    EVAL_ONLY=true ;;
    --skip-prep)    SKIP_PREP=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------- [0/6] env prep

if ! $SKIP_PREP; then
    echo "[0/6] Environment prep (venv + models) — see scripts/prep_env.sh"
    bash scripts/prep_env.sh
else
    echo "[0/6] --skip-prep: assuming env + models already ready"
fi

# Same env vars prep_env.sh exports — needed here for the training subprocesses too.
# Default to OFFLINE mode: models must be on disk already (see scripts/download_models.sh).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------- resolve model paths

# Locate the flat model dirs (prep_env.sh already downloaded them; here we just look them up).
find_local_model() {
    local repo_id="$1"
    local flat_name="$2"
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
    echo ""
}

QWEN_LOCAL="$(find_local_model "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b")"
E5_LOCAL="$(find_local_model "intfloat/e5-small-v2" "e5-small-v2")"
if [ -z "$QWEN_LOCAL" ] || [ -z "$E5_LOCAL" ]; then
    echo "❌ Missing model(s) after prep. Re-run scripts/prep_env.sh --models-only."
    echo "   Qwen3-4B → ${QWEN_LOCAL:-<not found>}"
    echo "   e5-small-v2 → ${E5_LOCAL:-<not found>}"
    exit 2
fi
echo "  ↳ Qwen3-4B → $QWEN_LOCAL"
echo "  ↳ e5-small-v2 → $E5_LOCAL"

# Rewrite HF repo ids in configs → local snapshot paths. This bypasses HF's cache-id lookup
# entirely, which is the safest way to load from a flat local dir.
GENERATED_CONFIG_DIR="outputs/generated_configs"
mkdir -p "$GENERATED_CONFIG_DIR"
for cfg in paper_grpo_answer_h100_qwen3_4b.yaml paper_grpo_manager_h100_qwen3_4b.yaml eval_h100_qwen3_4b_no_openai.yaml; do
    sed "s|Qwen/Qwen3-4B-Instruct-2507|$QWEN_LOCAL|g" "configs/$cfg" > "$GENERATED_CONFIG_DIR/$cfg"
done
echo "  ↳ runtime configs at $GENERATED_CONFIG_DIR/"

# ---------------------------------------------------------------------- Python runner

if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
    PY_RUN="uv run --no-sync"
elif [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY_RUN="python"
else
    PY_RUN="python"
fi

# Sanity: memory_r1 must be importable.
if ! $PY_RUN -c "import memory_r1" 2>/dev/null; then
    echo "  ↳ memory_r1 not importable; running 'pip install -e .' ..."
    $PY_RUN -m pip install -e . 2>&1 | tail -3
fi
echo "✓ memory_r1 importable"

# ---------------------------------------------------------------------- pipeline

if [ "$EVAL_ONLY" = "true" ]; then
    echo "[eval-only] Skipping data prep + training. Jumping to evaluation."
else
    # ---------------------------------------------------------------------- 1. data prep
    echo "[1/6] Manager training tuples (Algorithm 1)"
    if [ -f data/processed/manager_train.jsonl ] && [ -f data/processed/locomo_train.jsonl ]; then
        echo "  ✓ data/processed/manager_train.jsonl present — skipping."
    else
        $PY_RUN scripts/prepare_manager_data.py \
            --locomo data/raw/locomo/locomo10.json \
            --out data/processed/manager_train.jsonl \
            --extractor heuristic \
            --write-splits
    fi

    echo "[2/6] Answer Agent training tuples (Algorithm 2)"
    if [ -f data/processed/answer_train.jsonl ]; then
        echo "  ✓ data/processed/answer_train.jsonl present — skipping."
    else
        $PY_RUN scripts/prepare_answer_data.py \
            --locomo data/raw/locomo/locomo10.json \
            --out data/processed/answer_train.jsonl \
            --extractor heuristic \
            --encoder "$E5_LOCAL" \
            --top-k-per-speaker 30
    fi

    # ---------------------------------------------------------------------- 3. Stage A
    echo "[3/6] Stage A: Answer Agent GRPO (200 steps, batch=128, G=8)"
    STAGE_A_CKPT="outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
    if [ -d "$STAGE_A_CKPT" ]; then
        echo "  ✓ Stage A checkpoint at $STAGE_A_CKPT — skipping."
    else
        $PY_RUN scripts/train_answer_agent.py \
            outputs/generated_configs/paper_grpo_answer_h100_qwen3_4b.yaml
    fi

    if [ "$STAGE_A_ONLY" = "true" ]; then
        echo "[stage-a-only] Stopping after Stage A."
        exit 0
    fi

    # ---------------------------------------------------------------------- 4. Stage B
    echo "[4/6] Stage B: Memory Manager GRPO (200 steps)"
    STAGE_B_CKPT="outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
    if [ -d "$STAGE_B_CKPT" ]; then
        echo "  ✓ Stage B checkpoint at $STAGE_B_CKPT — skipping."
    else
        $PY_RUN scripts/train_memory_manager.py \
            outputs/generated_configs/paper_grpo_manager_h100_qwen3_4b.yaml
    fi
fi

# ---------------------------------------------------------------------- 5. eval
echo "[5/6] Evaluating on LoCoMo test set (1307 QA)"
$PY_RUN scripts/evaluate.py outputs/generated_configs/eval_h100_qwen3_4b_no_openai.yaml

echo
echo "🏁 Full pipeline complete."
echo "   Answer Agent checkpoint:   outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
echo "   Memory Manager checkpoint: outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
echo "   Predictions:               outputs/predictions_qwen3_4b_no_openai/"
