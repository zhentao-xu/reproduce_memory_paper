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

# By default we DON'T force HF offline mode — the H100 box has HF Hub access via the corp proxy,
# so we let the pipeline fetch missing models on-demand. Override with `HF_HUB_OFFLINE=1
# bash scripts/run_h100_qwen3_4b_pipeline.sh` to force offline mode.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Use ABSOLUTE paths for HF cache — relative './models' doesn't always resolve consistently
# across sentence-transformers / transformers / huggingface_hub subprocesses.
export HF_HOME="${HF_HOME:-$REPO_ROOT/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REPO_ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$REPO_ROOT/models}"  # legacy name
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

check_hf_cache() {
    # HF cache format: models--<org>--<model>/snapshots/<hash>/config.json (+ weights).
    # Missing snapshots/ typically means the user ran ``huggingface-cli download --local-dir``
    # (which writes flat files instead of the cache-format tree). Loading such a dir requires
    # the local path directly, not the HF repo id.
    local model_dir="$1"
    local repo_id="$2"
    if [ ! -d "$model_dir" ]; then
        echo
        echo "❌ Missing prerequisite: $model_dir"
        echo "   Download $repo_id on an online box:"
        echo "     HF_HOME=\$REPO_ROOT/models huggingface-cli download $repo_id"
        echo "   then scp/rsync the whole $model_dir to this host."
        exit 2
    fi
    if [ ! -d "$model_dir/snapshots" ]; then
        echo
        echo "❌ Corrupt HF cache: $model_dir has no snapshots/ subdir."
        echo "   Expected layout: $model_dir/snapshots/<hash>/config.json"
        echo "   Contents found:"
        ls -la "$model_dir" | head -10
        echo "   Re-download on an online box with:"
        echo "     HF_HOME=\$REPO_ROOT/models huggingface-cli download $repo_id"
        exit 2
    fi
    local snapshot_count
    snapshot_count=$(ls -1 "$model_dir/snapshots" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$snapshot_count" -eq 0 ]; then
        echo "❌ $model_dir/snapshots/ is empty. Re-download the model."
        exit 2
    fi
    echo "✓ $model_dir (with $snapshot_count snapshot(s))"
}

# Prefer a flat local-dir layout if present (from `snapshot_download(local_dir=..., local_dir_use_symlinks=False)`),
# fall back to the HF cache layout. Both are common depending on how the user downloaded.
find_local_model() {
    local repo_id="$1"     # e.g. Qwen/Qwen3-4B-Instruct-2507
    local flat_name="$2"   # e.g. qwen3-4b — where the user might have unpacked a flat local_dir download
    # Preferred: flat local_dir with an obvious name
    if [ -f "models/$flat_name/config.json" ]; then
        (cd "models/$flat_name" && pwd)
        return
    fi
    # Fallback 1: HF cache layout (models--<org>--<model>/snapshots/<hash>/config.json)
    local cache_dir_name="models--$(echo "$repo_id" | tr '/' '-' | sed 's/-/--/')"
    local cache_dir="models/$cache_dir_name"
    if [ -d "$cache_dir/snapshots" ]; then
        local snap_hash
        snap_hash="$(cat "$cache_dir/refs/main" 2>/dev/null || ls -1 "$cache_dir/snapshots" 2>/dev/null | head -1)"
        if [ -n "$snap_hash" ] && [ -f "$cache_dir/snapshots/$snap_hash/config.json" ]; then
            (cd "$cache_dir/snapshots/$snap_hash" && pwd)
            return
        fi
    fi
    # Fallback 2: flat under the org--name path without --
    if [ -f "models/$(basename "$repo_id")/config.json" ]; then
        (cd "models/$(basename "$repo_id")" && pwd)
        return
    fi
    echo ""
}

# If a flat model dir is present but the HF cache-format tree is broken/empty, create a
# symlink so both layouts work. This lets sentence-transformers / transformers find the model
# via the HF repo id AND via the direct path.
link_cache_to_flat() {
    local flat_dir="$1"                        # absolute path to the flat local dir
    local repo_id="$2"                         # e.g. intfloat/e5-small-v2
    local cache_name="models--$(echo "$repo_id" | tr '/' '-' | sed 's|-\([^-]\+\)$|--\1|')"
    # Note the sed above only replaces the LAST '-' — e.g.
    #   intfloat/e5-small-v2 → models--intfloat--e5-small-v2 (wrong, e5 has 3 dashes)
    # so we build it manually instead:
    local org="${repo_id%%/*}"
    local name="${repo_id#*/}"
    cache_name="models--${org}--${name}"
    local cache_dir="models/$cache_name"
    local snap_hash="localdir"
    local snap_dir="$cache_dir/snapshots/$snap_hash"

    mkdir -p "$cache_dir/refs" "$cache_dir/snapshots" "$cache_dir/blobs"
    echo "$snap_hash" > "$cache_dir/refs/main"
    # Nuke any dangling empty snapshot dir and point ours at the flat local dir.
    rm -rf "$snap_dir"
    ln -sfn "$flat_dir" "$snap_dir"
    echo "  ↳ linked $cache_dir/snapshots/$snap_hash → $flat_dir"
}

ensure_model() {
    # Verify the model is present locally; download from HF Hub into a flat local dir if not.
    # Flat dirs (regular files, no symlinks) are safe to copy between machines.
    local repo_id="$1"
    local flat_name="$2"
    local flat_dir="models/$flat_name"

    local resolved
    resolved="$(find_local_model "$repo_id" "$flat_name")"
    if [ -n "$resolved" ]; then
        echo "$resolved"
        return
    fi

    echo "  ↓ $repo_id not found locally — downloading to $flat_dir (HF_HUB_OFFLINE=$HF_HUB_OFFLINE)..." >&2
    if ! $PY_RUN -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$repo_id', local_dir='$flat_dir', local_dir_use_symlinks=False)
print('downloaded to:', p)
" >&2; then
        echo "❌ HF snapshot_download failed for $repo_id. If your box has no internet, either:" >&2
        echo "   - unset HF_HUB_OFFLINE=1 and retry (if HF is reachable via a proxy)" >&2
        echo "   - download on an online machine and rsync ./models/$flat_name over" >&2
        return
    fi

    resolved="$(find_local_model "$repo_id" "$flat_name")"
    echo "$resolved"
}

# We need $PY_RUN set BEFORE ensure_model. Move the runner detection up.
if command -v uv >/dev/null 2>&1 && [ -d .venv ]; then
    PY_RUN="uv run --no-sync"
elif [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY_RUN="python"
else
    PY_RUN="python"
fi

QWEN_LOCAL="$(ensure_model "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b")"
E5_LOCAL="$(ensure_model "intfloat/e5-small-v2" "e5-small-v2")"

# If we resolved via a flat dir, mirror it into the HF cache tree via a symlink.
if [ -n "$QWEN_LOCAL" ] && [ ! -f "models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/localdir/config.json" ]; then
    link_cache_to_flat "$QWEN_LOCAL" "Qwen/Qwen3-4B-Instruct-2507"
fi
if [ -n "$E5_LOCAL" ] && [ ! -f "models/models--intfloat--e5-small-v2/snapshots/localdir/config.json" ]; then
    link_cache_to_flat "$E5_LOCAL" "intfloat/e5-small-v2"
fi

# Some corp environments break HF's cache-id resolution (e.g. broken symlinks after rsync,
# or an older huggingface_hub that misreads HF_HUB_CACHE). To dodge that entirely we resolve
# each model to its local snapshot path and hand THAT to the training scripts. Local-path
# loads don't touch huggingface_hub at all.
resolve_local_snapshot() {
    local model_dir="$1"
    # Prefer refs/main; fall back to the first (usually only) snapshot dir if refs is empty.
    local snapshot_hash
    snapshot_hash="$(cat "$model_dir/refs/main" 2>/dev/null || true)"
    if [ -z "$snapshot_hash" ] || [ ! -d "$model_dir/snapshots/$snapshot_hash" ]; then
        snapshot_hash="$(ls -1 "$model_dir/snapshots" 2>/dev/null | head -1)"
    fi
    if [ -z "$snapshot_hash" ]; then
        echo ""
        return
    fi
    local snapshot_path="$model_dir/snapshots/$snapshot_hash"
    if [ ! -f "$snapshot_path/config.json" ]; then
        echo ""
        return
    fi
    # Convert to absolute path.
    (cd "$snapshot_path" && pwd)
}

diagnose_snapshot() {
    local model_dir="$1"
    echo
    echo "=== Diagnostic for $model_dir ==="
    if [ -f "$model_dir/refs/main" ]; then
        echo "  refs/main content: '$(cat "$model_dir/refs/main")'"
    else
        echo "  refs/main NOT FOUND"
    fi
    echo "  snapshots/ dirs:"
    ls -la "$model_dir/snapshots/" 2>&1 | sed 's/^/    /'
    for snap in "$model_dir/snapshots"/*; do
        [ -d "$snap" ] || continue
        echo "  Contents of $(basename "$snap"):"
        ls -la "$snap/" 2>&1 | sed 's/^/    /'
        break
    done
    echo "=========================================="
}

# This second block was the old blocking check — now handled by ensure_model() above which
# auto-downloads from HF Hub. Kept diagnose_snapshot as an available helper.
if [ -z "$QWEN_LOCAL" ] || [ -z "$E5_LOCAL" ]; then
    echo "❌ ensure_model failed — auto-download from HF Hub did not produce a valid model dir."
    echo "   Qwen3-4B → ${QWEN_LOCAL:-<not found>}"
    echo "   e5-small-v2 → ${E5_LOCAL:-<not found>}"
    echo "   Set HF_HUB_OFFLINE=0 explicitly if the corp box needs it, or manually download:"
    echo "     python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='models/qwen3-4b', local_dir_use_symlinks=False)\""
    echo "     python -c \"from huggingface_hub import snapshot_download; snapshot_download('intfloat/e5-small-v2', local_dir='models/e5-small-v2', local_dir_use_symlinks=False)\""
    exit 2
fi
echo "  ↳ resolved Qwen3-4B → $QWEN_LOCAL"
echo "  ↳ resolved e5-small-v2 → $E5_LOCAL"

# Generate runtime configs where all HF repo ids are replaced with the local snapshot paths.
# This avoids HF cache-id resolution entirely, working even when refs/ or symlinks are broken.
GENERATED_CONFIG_DIR="outputs/generated_configs"
mkdir -p "$GENERATED_CONFIG_DIR"
for cfg in paper_grpo_answer_h100_qwen3_4b.yaml paper_grpo_manager_h100_qwen3_4b.yaml eval_h100_qwen3_4b_no_openai.yaml; do
    sed "s|Qwen/Qwen3-4B-Instruct-2507|$QWEN_LOCAL|g" "configs/$cfg" > "$GENERATED_CONFIG_DIR/$cfg"
done
echo "  ↳ runtime configs at $GENERATED_CONFIG_DIR/ (repo ids → local paths)"

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
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"

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
            --encoder "$E5_LOCAL" \
            --top-k-per-speaker 30
    fi

    # ---------------------------------------------------------------------- 3. Stage A

    echo "[3/6] Stage A: RL fine-tuning the Answer Agent (200 steps, batch=128, G=8)..."
    STAGE_A_CKPT="outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
    if [ -d "$STAGE_A_CKPT" ]; then
        echo "  ✓ Stage A checkpoint already at $STAGE_A_CKPT. Skipping training."
    else
        $PY_RUN scripts/train_answer_agent.py \
            outputs/generated_configs/paper_grpo_answer_h100_qwen3_4b.yaml
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
            outputs/generated_configs/paper_grpo_manager_h100_qwen3_4b.yaml
    fi
fi

# ---------------------------------------------------------------------- 5. eval

echo "[5/6] Evaluating on LoCoMo test set (1307 QA)..."
$PY_RUN scripts/evaluate.py outputs/generated_configs/eval_h100_qwen3_4b_no_openai.yaml

echo
echo "🏁 Full pipeline complete."
echo "   Answer Agent checkpoint: outputs/checkpoints/paper_grpo_answer_qwen3_4b/step_200"
echo "   Memory Manager checkpoint: outputs/checkpoints/paper_grpo_manager_qwen3_4b/step_200"
echo "   Predictions:  outputs/predictions_qwen3_4b_no_openai/"
