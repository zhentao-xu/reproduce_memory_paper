#!/usr/bin/env bash
# =============================================================================
#  run_h100_qwen3_4b_pipeline.sh — full Memory-R1 reproduction pipeline.
# =============================================================================
#
#  WHAT IT DOES (in plain English):
#    Runs the entire Memory-R1 paper reproduction end-to-end. Concretely:
#      1. Turns the LoCoMo conversations into two supervised training sets
#         (one for the Memory Manager, one for the Answer Agent).
#      2. Fine-tunes the Answer Agent with GRPO (a reinforcement-learning
#         algorithm) so it learns to answer questions from retrieved memories.
#      3. Fine-tunes the Memory Manager with GRPO so it learns which memories
#         to update / add / delete after each conversation turn.
#      4. Evaluates the trained pair on the LoCoMo test set (1307 questions)
#         and writes predictions + metrics to `outputs/`.
#
#  WHAT IT NEEDS BEFORE YOU RUN IT:
#    - A working `.venv/` and installed dependencies. Get these by running
#      `bash scripts/prep_env.sh` first (only needs to happen once per checkout).
#    - The Qwen3-4B and e5-small-v2 model directories under `models/`.
#      `prep_env.sh` verifies these are present.
#    - A GPU. On an H100 (80 GB), the wall-clock estimates below are realistic.
#      On smaller GPUs (A100 40 GB / L40S) you may need to reduce batch size
#      in the config files.
#
#  WALL-CLOCK ESTIMATES on 1× H100 (80 GB):
#    Data prep    (heuristic extractor + e5-small-v2):    ~  5 min
#    Stage A:     Answer Agent GRPO   (200 steps):        ~  3-5 hours
#    Stage B:     Memory Manager GRPO (200 steps):        ~  5-7 hours
#    End-to-end eval on the 1307-question LoCoMo test:    ~ 30-45 min
#
#  USAGE (always run prep_env.sh first):
#    bash scripts/run_h100_qwen3_4b_pipeline.sh                  # full pipeline
#    bash scripts/run_h100_qwen3_4b_pipeline.sh --stage-a-only   # only Answer Agent training
#    bash scripts/run_h100_qwen3_4b_pipeline.sh --eval-only      # only re-run evaluation
#                                                                # (assumes checkpoints exist)
#
#  IDEMPOTENCY:
#    Every step checks whether its output already exists (a data file, a
#    checkpoint dir) and skips itself if so. Safe to Ctrl-C mid-run and re-run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

STAGE_A_ONLY=false
EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --stage-a-only) STAGE_A_ONLY=true ;;
    --eval-only)    EVAL_ONLY=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- HF-offline env vars
# Same env vars prep_env.sh set — we re-export them here because prep_env.sh
# runs in a separate shell so its exports don't propagate to us. These tell
# HuggingFace to load models from `models/` on disk and never hit the network.
# Also enable `PYTHONUNBUFFERED=1` so training logs stream in real time (no
# blocking buffering that hides progress for 30 seconds at a time).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"
export PYTHONUNBUFFERED=1
# Reduce CUDA allocator fragmentation. With G=8 candidates × ~4k-token sequences, each step
# frees and re-allocates large activation buffers; expandable_segments lets the allocator
# grow blocks instead of failing at first non-contiguous free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------- resolve model paths
# The config YAMLs shipped with the repo use HuggingFace repo IDs like
# "Qwen/Qwen3-4B-Instruct-2507". At runtime we want the training code to load
# from the LOCAL directory (offline). This block:
#   1. Finds where each model lives on disk (a flat `models/qwen3-4b/` dir OR
#      a HF cache dir with `snapshots/<hash>/`), and
#   2. Rewrites the YAML configs into `outputs/generated_configs/` with the HF
#      repo IDs replaced by absolute local paths. The training scripts then
#      load those generated configs.

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
# Pick which python to invoke: the `.venv` created by prep_env.sh if present
# (which is the expected case), else the system `python3`. `PY_RUN` is used
# for every training/eval subprocess below.

if [ -d .venv ] && [ -x .venv/bin/python ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY_RUN="python"
elif command -v python3 >/dev/null 2>&1; then
    PY_RUN="python3"
else
    PY_RUN="python"
fi

# Quick sanity check that prep_env.sh actually ran successfully — the training subprocesses
# below all `import memory_r1`, so failing here surfaces the problem in 1s instead of 30s.
if ! $PY_RUN -c "import memory_r1" 2>/dev/null; then
    echo "❌ 'memory_r1' Python package not importable."
    echo "   Run 'bash scripts/prep_env.sh' first (it creates .venv and installs the package)."
    exit 4
fi
echo "✓ memory_r1 importable"

# ---------------------------------------------------------------------- pre-flight banner
# Print a one-shot summary of everything the pipeline is about to use — Python
# version, PyTorch version, whether CUDA is visible, resolved model paths, etc.
# Useful when things go wrong (paste the banner into a bug report; it tells
# you 90 % of what someone would ask).

echo
echo "=================== PRE-FLIGHT SUMMARY ==================="
echo "  Working dir:  $REPO_ROOT"
echo "  Python:       $($PY_RUN -c 'import sys; print(sys.version.split()[0])')"
echo "  Torch:        $($PY_RUN -c 'import torch; print(torch.__version__)')"
echo "  CUDA:         $($PY_RUN -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), "device(s)")')"
echo "  HF offline:   $HF_HUB_OFFLINE"
echo "  Qwen model:   $QWEN_LOCAL"
echo "  e5 model:     $E5_LOCAL"
echo "  Config dir:   $GENERATED_CONFIG_DIR/"
echo "  Stages:       $( [ "$EVAL_ONLY" = "true" ] && echo "eval-only" || ( [ "$STAGE_A_ONLY" = "true" ] && echo "prep+stageA" || echo "prep+stageA+stageB+eval" ) )"
echo "=========================================================="
echo

# ---------------------------------------------------------------------- pipeline

if [ "$EVAL_ONLY" = "true" ]; then
    echo "[eval-only] Skipping data prep + training. Jumping to evaluation."
else
    # ---------------------------------------------------------------------- 1. data prep
    # Turn raw LoCoMo conversations into training tuples for the Memory Manager.
    # Each tuple = (retrieved_facts, old_memory_bank, linked_QA). The Manager
    # will be RL-trained to produce ADD/UPDATE/DELETE/NOOP operations against
    # the old memory bank such that the downstream Answer Agent gets the linked
    # QA right. This step ships alongside `--write-splits` so a train/val split
    # of the raw conversations is also produced (used downstream by eval).
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

    # Turn raw LoCoMo conversations into training tuples for the Answer Agent.
    # Each tuple = (question, retrieved_top_60_memories, gold_answer). "top-60"
    # means top-30 memories per speaker (there are always 2 speakers in a LoCoMo
    # dialogue), retrieved by e5-small-v2 cosine similarity. The Answer Agent
    # is trained to produce the gold answer given those 60 memories as context.
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
    # GRPO fine-tune the Answer Agent on the tuples produced in step [2/6].
    # GRPO = Group-Relative Policy Optimization: for each prompt we sample G=8
    # candidate answers, reward each against the gold answer, and use the
    # group-normalized advantages to update the LoRA adapters. Paper hparams:
    # 200 steps, batch size 128, group size 8. Wall-clock: 3-5 hours on 1× H100.
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
    # GRPO fine-tune the Memory Manager. The reward loop is: for each Manager
    # rollout (a proposed set of ADD/UPDATE/DELETE/NOOP ops), APPLY the ops to
    # the memory bank, then feed the resulting bank to the frozen Stage-A
    # Answer Agent and score whether it now answers the linked question
    # correctly. The Manager therefore learns to curate memory in a way that
    # actively helps downstream QA. Wall-clock: 5-7 hours on 1× H100.
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
# Run the fully-trained (Manager, Answer Agent) pair on the LoCoMo test set:
# 10 held-out conversations × ~130 QA each = 1307 questions. Metrics reported:
# Exact-Match, token-level F1, BLEU-1, and (if `judge` reward configured) an
# LLM-judge accuracy. Predictions land in outputs/predictions_qwen3_4b_no_openai/.
echo "[5/6] Evaluating on LoCoMo test set (1307 QA)"
$PY_RUN scripts/evaluate.py outputs/generated_configs/eval_h100_qwen3_4b_no_openai.yaml

echo
echo "🏁 Full pipeline complete."
echo "   Answer Agent checkpoint:   outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
echo "   Memory Manager checkpoint: outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
echo "   Predictions:               outputs/predictions_qwen3_4b_no_openai/"
