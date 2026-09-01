# Memory-R1 (Reproduction)

End-to-end reproduction of **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via
Reinforcement Learning** (Yan et al., arXiv:2508.19828, v5, 2026-01-14).

The paper proposes two RL-fine-tuned agents:

1. **Memory Manager** — chooses `ADD / UPDATE / DELETE / NOOP` for each new dialogue turn given the
   current memory bank. Trained with PPO or GRPO. Reward comes from the *frozen Answer Agent*'s
   exact-match against the gold answer.
2. **Answer Agent** — takes 60 retrieved memories (30 per speaker) and does *Memory Distillation*:
   selects the relevant subset and produces a concise answer. Trained with PPO or GRPO. Reward is
   direct EM against gold.

We reproduce the entire pipeline: data construction (Algorithms 1 & 2), the two RL trainers, and
end-to-end evaluation on **LoCoMo**, **MSC**, and **LongMemEval** — the same three benchmarks the
paper uses.

Only 152 training QA pairs are used, following the paper's 1:1:8 LoCoMo split.

---

## 1. Setup with `uv`

```bash
uv sync                            # install everything (torch, transformers, trl, faiss, ...)
uv run pytest -q                   # sanity-check the pure-python parts
```

Copy `.env.example → .env`. The default `.env` redirects HuggingFace model downloads into
`./models/` (a local sibling of `./data/`) rather than the global `~/.cache/huggingface` cache —
keeps the project self-contained. Fill in `OPENAI_API_KEY` for GPT-4o-mini extraction / judge and
`HUGGINGFACE_HUB_TOKEN` for gated models.

```bash
cp .env.example .env
set -a; source .env; set +a       # load into current shell
```

Repository layout after setup:

```
data/       # datasets (LoCoMo, MSC, LongMemEval) + processed JSONLs
models/     # HuggingFace model cache (redirected via HF_HOME)
outputs/    # RL checkpoints + predictions
logs/       # per-execution run.log / run.jsonl; ``latest`` symlinks to newest run
```

Every script writes a colored `run.log` and structured `run.jsonl` into
`logs/<execution_id>/`. `logs/latest` always points at the most recent invocation, so in
another terminal you can watch live:

```bash
tail -F logs/latest/run.log            # plain: ANSI codes visible
less -R logs/latest/run.log            # colored: per-level shading
cat logs/latest/status.json 2>/dev/null # trainer's current step (updated every iteration)
```

Logging emoji palette used in code (all hard-coded in the log strings):

- 🚀 start   🏁 done   ⚙️  config   📥 load/download   🧠 model   🖥️ device
- 📚 data    🏋️  train  🎲 rollout   🎯 reward          📉 grad    🔄 step
- 📊 eval    💾 save   ✅ ok        ⚠️  warn           ❌ error

---

## 2. Download the three benchmarks

```bash
uv run python scripts/download_datasets.py --dataset all
```

This populates:

- `data/raw/locomo/locomo10.json` — 10 dialogues, ~300 turns each, categories 1..5 (we drop 5).
- `data/raw/msc/msc.jsonl`         — Multi-Session Chat (memory-eval variant, per MemGPT).
- `data/raw/longmemeval/longmemeval_s.json` — SSU/SSP/OD/MS/KU/TR question types.

If HF or the ParlAI mirror is down, the script prints a helpful fallback pointing to the original
release page.

---

## 3. Data construction (Algorithms 1 & 2)

Memory Manager tuples (`(dialogue_turn, temporal_memory_bank, QA)`) — paper's Algorithm 1:

```bash
uv run python scripts/prepare_manager_data.py \
    --locomo data/raw/locomo/locomo10.json \
    --out    data/processed/manager_train.jsonl \
    --extractor openai --model gpt-4o-mini \
    --write-splits
```

Answer Agent tuples (`(question, retrieved_60, gold)`) — paper's Algorithm 2:

```bash
uv run python scripts/prepare_answer_data.py \
    --locomo data/raw/locomo/locomo10.json \
    --out    data/processed/answer_train.jsonl \
    --extractor openai --model gpt-4o-mini \
    --manager-backend openai --manager-model gpt-4o-mini \
    --top-k-per-speaker 30
```

Tips:

- Drop `--extractor openai` to use the offline heuristic extractor (splits utterances into
  sentences). Good for smoke tests; not paper-faithful.
- `--max-dialogues 1` runs on a single dialogue for quick iteration.

---

## 4. RL fine-tuning

Two agents × two RL algorithms → four configs under `configs/`.

Config names follow `<stage>_<algo>_<device>_<model>.yaml`.

**Memory Manager (GRPO / LLaMA-3.1-8B-Instruct on 1× H100)**

```bash
uv run python scripts/train_memory_manager.py configs/manager_grpo_h100_llama_8b.yaml
```

**Answer Agent (GRPO / LLaMA-3.1-8B-Instruct on 1× H100)**

```bash
uv run python scripts/train_answer_agent.py configs/answer_grpo_h100_llama_8b.yaml
```

PPO variants: `configs/manager_ppo_h100_llama_8b.yaml`, `configs/answer_ppo_h100_llama_8b.yaml`.
Qwen backbone (7B): `configs/manager_grpo_h100_qwen_7b.yaml`, `configs/answer_grpo_h100_qwen_7b.yaml`.

Hyperparameters match paper Appendix D:

| Parameter                       | Value            | Source           |
| ------------------------------- | ---------------- | ---------------- |
| Actor lr                        | 1e-6             | Appendix D       |
| Critic lr (PPO)                 | 1e-5             | Appendix D       |
| Total batch                     | 128              | Appendix D       |
| Micro-batch / GPU               | 2                | Appendix D       |
| Max prompt / response length    | 4096 / 2048      | Appendix D       |
| Training temperature (rollout)  | 1.0              | Appendix D       |
| Eval temperature                | 0.0 (greedy)     | Appendix D       |
| Total training steps            | 200              | Figure 7         |
| Reward                          | EM against gold  | Section 3.1/3.2  |
| Retrieval K per speaker (Answer)| 30 (60 total)    | Section 3.2      |

The trainer supports optional LoRA (`use_peft: true`) so you can fit training on a single
40-GB GPU when 4×H100 aren't available.

**Local testing with Qwen3-4B on Apple Silicon.** For quick iteration on an M-series Mac, use the
smaller `configs/manager_grpo_mps_qwen3_4b.yaml` / `configs/answer_grpo_mps_qwen3_4b.yaml` with
`Qwen/Qwen3-4B-Instruct-2507`. The trainer auto-detects MPS via `resolve_device()` — no manual flags
needed. Force a specific device with `MEMORY_R1_DEVICE=mps|cuda|cpu`.

```bash
MEMORY_R1_DEVICE=mps uv run python scripts/train_answer_agent.py \
    configs/answer_grpo_mps_qwen3_4b.yaml
```

`configs/smoke_grpo_answer_qwen3_4b.yaml` runs 2 steps in ~1 min on M2 Max — useful for verifying
the whole pipeline (rollout → reward → PPO/GRPO update → checkpoint save) before spending real GPU
budget.

---

## 5. End-to-end evaluation

Pick the eval config matching the run you just trained:

```bash
uv run python scripts/evaluate.py configs/eval_h100_llama_8b.yaml     # LLaMA-8B on H100
uv run python scripts/evaluate.py configs/eval_h100_qwen_7b.yaml      # Qwen-7B on H100
uv run python scripts/evaluate.py configs/eval_mps_qwen3_4b.yaml      # Qwen3-4B on M-series MPS
```

Prints a table of F1 / BLEU-1 / EM / LLM-as-a-Judge (J) broken down by question type
(`single_hop`, `multi_hop`, `open_domain`, `temporal`) and saves predictions + summary under
`outputs/predictions_<device>_<model>/`.

To reproduce the paper's baseline row (no RL), use `configs/eval_base_h100_llama_8b.yaml`.

---

## 6. Repository layout

```
src/memory_r1/
├── data/            # LoCoMo/MSC/LongMemEval loaders + Algorithms 1-2
├── memory/          # MemoryBank, ADD/UPDATE/DELETE/NOOP, dense retriever
├── agents/          # FactExtractor, MemoryManager, AnswerAgent, LLM backends
├── prompts/         # Verbatim prompts from paper Figures 9-12
├── training/        # PPO + GRPO trainers, reward pipeline, configs
├── eval/            # F1 / BLEU-1 / LLM-as-a-Judge + evaluator
└── cli.py           # `uv run memory-r1 ...` entrypoint

configs/             # YAML configs for train + eval
scripts/             # Direct-python entrypoints
tests/               # Unit tests (memory/metrics/prompts/answer parsing)
paper/               # PDF of the reproduced paper
```

---

## 7. Deviations from the paper (worth knowing before you extend)

1. **RL library.** The paper uses VERL (Sheng et al., 2025). We implement equivalent PPO and GRPO
   objectives in ~300 lines each. Same reward, same rollout shape, same hyperparameters — but the
   underlying gradient update is our own PyTorch code, not VERL's. This is much easier to hack on.
2. **Answer-Agent reward inside Manager training.** During Manager RL, the paper feeds the updated
   bank to a *frozen* Answer Agent. Our default frozen Answer Agent is the base LLM
   (no RL). If you first train the Answer Agent and set `answer_backend.checkpoint`, the Manager is
   trained against the *RL-trained* Answer Agent — the paper's ideal setting.
3. **Fact extractor.** The paper uses GPT-4o-mini as `LLMExtract` (Algorithm 3). Our default is
   also `gpt-4o-mini` when you pass `--extractor openai`. A heuristic fallback splits utterances
   into sentences so the pipeline is runnable offline for CI.
4. **Encoder.** RAG uses `sentence-transformers/all-MiniLM-L6-v2` by default — same as Mem0, the
   baseline Memory-R1 compares against. Swap via `DenseRetriever(encoder_name=...)`.
5. **NONE vs NOOP.** The paper's *prompt* uses `NONE` (Figure 10); the paper's *algorithm text*
   uses `NOOP`. We keep both — the parser maps `NOOP` → `ManagerOp.NOOP` = `"NONE"`.

---

## 8. Extending

Common places to modify:

- **New reward** — implement `build_answer_reward_fn("myreward")` and add a branch in
  `reward_pipeline.py`.
- **Different backbone** — change `model.name_or_path` in the YAML. LLaMA/Qwen/Mistral chat
  templates are all supported via `tokenizer.apply_chat_template`.
- **Different memory encoding** — extend `MemoryBank` (e.g. per-entry embeddings, decay weights)
  and pass to `DenseRetriever`.
- **New RL objective** — subclass `GRPOManagerTrainer` / `PPOManagerTrainer` and override
  `_grpo_step` / `_ppo_step`.

---

## 9. Citation

If you build on this reproduction, please cite the original paper:

```bibtex
@article{yan2026memoryr1,
  title  = {Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via
            Reinforcement Learning},
  author = {Yan, Sikuan and Yang, Xiufeng and Huang, Zuchao and Nie, Ercong and Ding, Zifeng and
            Li, Zonggen and Ma, Xiaowen and Bi, Jinhe and Kersting, Kristian and Pan, Jeff Z. and
            Schuetze, Hinrich and Tresp, Volker and Ma, Yunpu},
  journal = {arXiv preprint arXiv:2508.19828},
  year   = {2026}
}
```

License: Apache-2.0 (this repo). LoCoMo is CC BY-NC 4.0 (research use only).
