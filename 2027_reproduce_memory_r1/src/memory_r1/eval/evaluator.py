"""End-to-end evaluator.

Given a YAML config, runs the full Memory-R1 pipeline over a benchmark:

1. For each dialogue, run the Memory Manager over every turn → build the memory bank.
2. For each question, retrieve 60 memories (30 per speaker) → run the Answer Agent → parse the
   final answer.
3. Compute F1, BLEU-1, and (optionally) LLM-as-a-Judge, both overall and broken down by question
   type (single_hop / multi_hop / open_domain / temporal for LoCoMo; SSU / SSP / OD / MS / KU / TR
   for LongMemEval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from memory_r1.agents.answer_agent import AnswerAgent
from memory_r1.agents.extractor import FactExtractor
from memory_r1.agents.llm_backend import HFBackend, LLMBackend, OpenAIBackend
from memory_r1.agents.memory_manager import MemoryManager
from memory_r1.data.locomo import LoCoMoLoader
from memory_r1.eval.judge import LLMJudge
from memory_r1.eval.metrics import bleu1, exact_match, token_f1
from memory_r1.memory.bank import MemoryBank
from memory_r1.memory.retrieval import DenseRetriever

console = Console()


# --------------------------------------------------------------------------- config


@dataclass
class EvalConfig:
    benchmark: str = "locomo"
    data_path: Path = Path("data/raw/locomo/locomo10.json")
    manager_checkpoint: Path | str = "meta-llama/Llama-3.1-8B-Instruct"
    answer_checkpoint: Path | str = "meta-llama/Llama-3.1-8B-Instruct"
    extractor_backend: str = "openai"  # or "hf"
    extractor_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    top_k_per_speaker: int = 30
    top_k_context_manager: int = 10
    max_dialogues: int | None = None
    max_questions_per_dialogue: int | None = None
    output_dir: Path = Path("outputs/predictions")
    use_judge: bool = True
    dtype: str = "bfloat16"
    max_new_tokens_manager: int = 2048
    max_new_tokens_answer: int = 512


def load_eval_config(path: str | Path) -> EvalConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    cfg = EvalConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.data_path = Path(cfg.data_path)
    cfg.output_dir = Path(cfg.output_dir)
    return cfg


# --------------------------------------------------------------------------- results


@dataclass
class QAResult:
    dialogue_id: str
    question: str
    gold: str
    pred: str
    category: str
    f1: float
    b1: float
    em: float
    judge: bool | None = None
    raw_answer: str = ""
    distilled: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- pipeline


def _make_extractor_backend(cfg: EvalConfig) -> LLMBackend:
    if cfg.extractor_backend == "openai":
        return OpenAIBackend(model=cfg.extractor_model)
    if cfg.extractor_backend == "hf":
        return HFBackend(model_name_or_path=cfg.extractor_model, dtype=cfg.dtype)
    raise ValueError(cfg.extractor_backend)


def _build_bank(
    dialogue,
    extractor: FactExtractor,
    manager: MemoryManager,
) -> MemoryBank:
    """Algorithm 3: run the Memory Manager over every turn."""

    bank = MemoryBank()
    for turn in dialogue.turns:
        facts = extractor.extract(turn.speaker, turn.text, turn.timestamp).facts
        if not facts:
            continue
        manager.step(bank, turn.speaker, facts, turn_timestamp=turn.timestamp)
    return bank


def run_evaluation(config_path: str | Path) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    console.rule(f"[bold]Evaluating on {cfg.benchmark}[/bold]")

    # Backbones.
    console.log(f"Loading Memory Manager from: {cfg.manager_checkpoint}")
    manager_backend = HFBackend(model_name_or_path=str(cfg.manager_checkpoint), dtype=cfg.dtype)
    console.log(f"Loading Answer Agent from: {cfg.answer_checkpoint}")
    answer_backend = HFBackend(model_name_or_path=str(cfg.answer_checkpoint), dtype=cfg.dtype)
    extractor_backend = _make_extractor_backend(cfg)

    retriever = DenseRetriever()
    extractor = FactExtractor(extractor_backend)
    manager = MemoryManager(
        backend=manager_backend,
        retriever=retriever,
        top_k_context=cfg.top_k_context_manager,
        temperature=0.0,
        max_tokens=cfg.max_new_tokens_manager,
    )
    answer = AnswerAgent(backend=answer_backend, temperature=0.0, max_tokens=cfg.max_new_tokens_answer)
    judge = LLMJudge(backend=OpenAIBackend(model=cfg.judge_model)) if cfg.use_judge else None

    # Data.
    if cfg.benchmark == "locomo":
        loader = LoCoMoLoader(cfg.data_path, exclude_adversarial=True)
        dialogues = loader.dialogues
    else:
        raise NotImplementedError(f"benchmark {cfg.benchmark!r} loader not implemented in evaluator")
    if cfg.max_dialogues:
        dialogues = dialogues[: cfg.max_dialogues]

    # Run.
    results: list[QAResult] = []
    for dialogue in track(dialogues, description="Building banks + answering"):
        bank = _build_bank(dialogue, extractor, manager)
        qas = dialogue.qa if cfg.max_questions_per_dialogue is None else dialogue.qa[: cfg.max_questions_per_dialogue]
        for qa in qas:
            retrieved = retriever.search_by_speaker(bank, qa.question, top_k_per_speaker=cfg.top_k_per_speaker)
            mem_by_speaker = {
                sp: [{"id": h.entry_id, "text": h.text, "timestamp": h.timestamp or ""} for h in hits]
                for sp, hits in retrieved.items()
            }
            out = answer.answer(question=qa.question, memories_by_speaker=mem_by_speaker)
            f1 = token_f1(out.answer, qa.answer)
            b1 = bleu1(out.answer, qa.answer)
            em = exact_match(out.answer, qa.answer)
            j = None
            if judge is not None:
                j = judge.judge(qa.question, qa.answer, out.answer).correct
            results.append(
                QAResult(
                    dialogue_id=dialogue.dialogue_id,
                    question=qa.question,
                    gold=qa.answer,
                    pred=out.answer,
                    category=qa.category_name,
                    f1=f1,
                    b1=b1,
                    em=em,
                    judge=j,
                    raw_answer=out.raw,
                    distilled=out.distilled_memories,
                )
            )

    # Persist predictions.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cfg.output_dir / f"{cfg.benchmark}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    console.log(f"[green]Predictions → {pred_path}[/green]")

    # Aggregate.
    summary = _summarize(results)
    _print_summary(summary)

    summary_path = cfg.output_dir / f"{cfg.benchmark}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    console.log(f"[green]Summary → {summary_path}[/green]")
    return summary


# --------------------------------------------------------------------------- reporting


def _summarize(results: list[QAResult]) -> dict[str, Any]:
    by_cat: dict[str, list[QAResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    out: dict[str, Any] = {"overall": _agg(results)}
    for cat, rs in by_cat.items():
        out[cat] = _agg(rs)
    return out


def _agg(rs: list[QAResult]) -> dict[str, float]:
    n = max(1, len(rs))
    j_scored = [r for r in rs if r.judge is not None]
    return {
        "n": len(rs),
        "F1": sum(r.f1 for r in rs) * 100 / n,
        "B1": sum(r.b1 for r in rs) * 100 / n,
        "EM": sum(r.em for r in rs) * 100 / n,
        "J": (sum(1 for r in j_scored if r.judge) * 100 / max(1, len(j_scored))) if j_scored else None,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Memory-R1 evaluation")
    table.add_column("Split")
    table.add_column("n", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("B1", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("J", justify="right")

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else ("--" if x is None else str(x))

    ordered = ["overall", "single_hop", "multi_hop", "open_domain", "temporal"]
    for key in ordered:
        if key not in summary:
            continue
        s = summary[key]
        table.add_row(key, str(s["n"]), fmt(s["F1"]), fmt(s["B1"]), fmt(s["EM"]), fmt(s["J"]))
    for key, s in summary.items():
        if key in ordered:
            continue
        table.add_row(key, str(s["n"]), fmt(s["F1"]), fmt(s["B1"]), fmt(s["EM"]), fmt(s["J"]))
    console.print(table)
