"""Top-level Typer CLI: ``uv run memory-r1 <subcommand>``.

Sub-commands are thin wrappers around scripts under ``scripts/`` so that everything can be run
without knowing the exact python module path.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Memory-R1 reproduction CLI.")


@app.command("download")
def download(
    dataset: str = typer.Option("all", help="Which dataset to download: locomo | msc | longmemeval | all"),
    out_dir: Path = typer.Option(Path("data/raw"), help="Destination for raw datasets."),
) -> None:
    """Download LoCoMo, MSC, LongMemEval into ``data/raw/``."""
    from memory_r1.data.download import download_all

    download_all(dataset=dataset, out_dir=out_dir)


@app.command("build-manager-data")
def build_manager_data(
    locomo_path: Path = typer.Option(Path("data/raw/locomo/locomo10.json")),
    out_path: Path = typer.Option(Path("data/processed/manager_train.jsonl")),
    lookback_turns: int = typer.Option(24, help="Turns of context used to build the temporal memory bank."),
    max_dialogues: int | None = typer.Option(None, help="Optional cap for a quick smoke test."),
) -> None:
    """Algorithm 1: build (dialogue turn, temporal memory bank, QA) tuples for the Memory Manager."""
    from memory_r1.data.construction import build_manager_dataset

    build_manager_dataset(
        locomo_path=locomo_path,
        out_path=out_path,
        lookback_turns=lookback_turns,
        max_dialogues=max_dialogues,
    )


@app.command("build-answer-data")
def build_answer_data(
    locomo_path: Path = typer.Option(Path("data/raw/locomo/locomo10.json")),
    out_path: Path = typer.Option(Path("data/processed/answer_train.jsonl")),
    top_k_per_speaker: int = typer.Option(30, help="Top-K per participant (60 total)."),
    max_dialogues: int | None = typer.Option(None),
) -> None:
    """Algorithm 2: build (question, 60 retrieved memories, gold) tuples for the Answer Agent."""
    from memory_r1.data.construction import build_answer_dataset

    build_answer_dataset(
        locomo_path=locomo_path,
        out_path=out_path,
        top_k_per_speaker=top_k_per_speaker,
        max_dialogues=max_dialogues,
    )


@app.command("train-manager")
def train_manager(
    config: Path = typer.Argument(..., help="YAML config, e.g. configs/grpo_manager.yaml"),
) -> None:
    """RL fine-tune the Memory Manager (PPO or GRPO, chosen inside the config)."""
    from memory_r1.training.entrypoints import train_memory_manager

    train_memory_manager(config)


@app.command("train-answer")
def train_answer(
    config: Path = typer.Argument(..., help="YAML config, e.g. configs/grpo_answer.yaml"),
) -> None:
    """RL fine-tune the Answer Agent (PPO or GRPO)."""
    from memory_r1.training.entrypoints import train_answer_agent

    train_answer_agent(config)


@app.command("evaluate")
def evaluate(
    config: Path = typer.Argument(..., help="YAML config, e.g. configs/eval_locomo.yaml"),
) -> None:
    """Run the full pipeline on a benchmark and report F1 / BLEU-1 / LLM-as-a-Judge."""
    from memory_r1.eval.evaluator import run_evaluation

    run_evaluation(config)


if __name__ == "__main__":
    app()
