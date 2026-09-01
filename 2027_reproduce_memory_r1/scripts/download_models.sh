#!/usr/bin/env bash
# =============================================================================
#  download_models.sh — fetch the HuggingFace models Memory-R1 needs.
# =============================================================================
#
#  Run on any box with HuggingFace Hub access (Mac laptop, jump host, ...), then
#  `rsync` the resulting `models/` dir to the offline H100.
#
#  Model bundles (pick one via `--models`):
#    small (DEFAULT) — the open-source-only reproduction (~8 GB):
#        models/qwen3-4b/       Qwen/Qwen3-4B-Instruct-2507       (~8 GB)
#        models/e5-small-v2/    intfloat/e5-small-v2              (~150 MB)
#
#    paper — the two backbones used in the paper's Table 1 (~30 GB):
#        models/llama-3.1-8b/   meta-llama/Llama-3.1-8B-Instruct  (~16 GB, GATED)
#        models/qwen2.5-7b/     Qwen/Qwen2.5-7B-Instruct          (~14 GB)
#        models/e5-small-v2/    intfloat/e5-small-v2              (~150 MB)
#
#    extractor — the local fact-extractor replacement for GPT-4o-mini (~54 GB):
#        models/qwen3.8-27b/    Qwen/Qwen3.8-27B                  (~54 GB, 18 shards)
#
#    all — everything above (~92 GB total).
#
#  LLaMA-3.1-8B is a GATED model on HuggingFace. Before downloading, you must:
#    1. Accept the LLaMA 3.1 license at
#       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#    2. Export a token in the shell you run this from:
#         export HUGGINGFACE_HUB_TOKEN=hf_...
#       or run `huggingface-cli login` once.
#
#  Every model dir is FLAT (no HF cache tree, no symlinks) so it's safe to
#  rsync/scp directly to any offline machine and load with
#  `AutoModel.from_pretrained("models/<name>")`.
#
#  USAGE:
#    bash scripts/download_models.sh                     # small bundle (default)
#    bash scripts/download_models.sh --models paper      # paper Table-1 backbones
#    bash scripts/download_models.sh --models extractor  # only the 27B fact extractor
#    bash scripts/download_models.sh --models all        # everything (~92 GB)
#
#  Then on your workstation:
#    rsync -av --info=progress2 models/ user@h100:/path/to/repo/models/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "  (working dir: $REPO_ROOT)"

# ---------------------------------------------------------------- args
BUNDLE="small"
for arg in "$@"; do
  case "$arg" in
    --models=*) BUNDLE="${arg#*=}" ;;
    --models)   BUNDLE="__NEXT__" ;;
    __NEXT__)   ;;  # placeholder — shouldn't happen
    *)
      # If preceding token was --models, treat this as its value.
      if [ "$BUNDLE" = "__NEXT__" ]; then
          BUNDLE="$arg"
      else
          echo "Unknown arg: $arg" >&2
          echo "Valid: --models {small|paper|extractor|all}" >&2
          exit 1
      fi
      ;;
  esac
done
case "$BUNDLE" in
  small|paper|extractor|all) ;;
  *) echo "Unknown bundle: $BUNDLE (valid: small, paper, extractor, all)" >&2; exit 1 ;;
esac
echo "  → bundle: $BUNDLE"

mkdir -p models

# ---------------------------------------------------------------- python + hf hub
# Prefer .venv, then uv, then system python. huggingface_hub must be importable.
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

# ---------------------------------------------------------------- LLaMA gating warning
if [ "$BUNDLE" = "paper" ] || [ "$BUNDLE" = "all" ]; then
    if [ -z "${HUGGINGFACE_HUB_TOKEN:-}${HF_TOKEN:-}" ] \
        && ! $PY -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
        echo
        echo "⚠  LLaMA-3.1-8B is GATED and no HF token is visible in this shell."
        echo "   Either:"
        echo "     1. Accept the license at"
        echo "        https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
        echo "        then run 'huggingface-cli login' (persists the token), OR"
        echo "     2. export HUGGINGFACE_HUB_TOKEN=hf_...  before re-running this script."
        echo "   Aborting so you don't hit a 401 mid-download."
        exit 1
    fi
fi

# ---------------------------------------------------------------- downloader
download_one() {
    local repo_id="$1"
    local flat_dir="$2"
    if [ -f "$flat_dir/config.json" ]; then
        echo "  ✓ $repo_id already at $flat_dir — skipping"
        return
    fi
    echo "  ↓ downloading $repo_id → $flat_dir"
    # Skip redundant weight formats (ONNX / TF / OpenVINO / legacy pytorch_model.bin).
    # safetensors is the format transformers + sentence-transformers load by default.
    $PY - <<PYEOF
from huggingface_hub import snapshot_download
ignore = [
    "*.onnx",
    "onnx/*", "onnx_*",
    "openvino/*", "openvino_*",
    "tf_model.h5",
    "flax_model.msgpack",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
]
p = snapshot_download("$repo_id", local_dir="$flat_dir", ignore_patterns=ignore)
print(f"  ✓ done: {p}")
PYEOF
}

# ---------------------------------------------------------------- dispatch
case "$BUNDLE" in
  small)
    download_one "Qwen/Qwen3-4B-Instruct-2507"     "models/qwen3-4b"
    download_one "intfloat/e5-small-v2"            "models/e5-small-v2"
    ;;
  paper)
    download_one "meta-llama/Llama-3.1-8B-Instruct" "models/llama-3.1-8b"
    download_one "Qwen/Qwen2.5-7B-Instruct"         "models/qwen2.5-7b"
    download_one "intfloat/e5-small-v2"             "models/e5-small-v2"
    ;;
  extractor)
    download_one "Qwen/Qwen3.8-27B"                 "models/qwen3.8-27b"
    ;;
  all)
    download_one "Qwen/Qwen3-4B-Instruct-2507"      "models/qwen3-4b"
    download_one "meta-llama/Llama-3.1-8B-Instruct" "models/llama-3.1-8b"
    download_one "Qwen/Qwen2.5-7B-Instruct"         "models/qwen2.5-7b"
    download_one "Qwen/Qwen3.8-27B"                 "models/qwen3.8-27b"
    download_one "intfloat/e5-small-v2"             "models/e5-small-v2"
    ;;
esac

echo
echo "✅ Models ready under $REPO_ROOT/models/"
echo
echo "Next: copy to the offline machine, e.g."
echo "  rsync -av --info=progress2 models/ user@h100:/path/to/repo/models/"
