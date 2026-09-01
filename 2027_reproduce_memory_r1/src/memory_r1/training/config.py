"""Typed YAML configs for training / evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    """Backbone policy model."""

    name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    use_peft: bool = False
    peft_r: int = 16
    peft_alpha: int = 32
    peft_dropout: float = 0.05
    # When true, actor runs with ``gradient_checkpointing_enable()`` — trades ~30 % compute
    # for ~5-10× activation-memory reduction. Default TRUE because at G=16 + µ=8 on 4B model,
    # storing all layer activations across all micro-batch candidates eats 30-40 GB per forward
    # and blows past even 80 GB H100. Only turn this off if you've measured VRAM headroom
    # explicitly (e.g. small G, short responses, or a smaller model).
    use_gradient_checkpointing: bool = True


@dataclass
class OptimConfig:
    """Optimizer / schedule per paper Appendix D."""

    actor_lr: float = 1e-6  # paper
    critic_lr: float = 1e-5  # paper (PPO only)
    total_batch_size: int = 128  # paper: total batch size 128
    micro_batch_size_per_gpu: int = 2  # paper: mb 2 per GPU
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.02
    clip_range: float = 0.2
    max_grad_norm: float = 1.0


@dataclass
class RLConfig:
    algorithm: Literal["ppo", "grpo"] = "grpo"
    # GRPO specifics (paper: group of G candidate actions per state).
    group_size: int = 8
    grpo_beta: float = 0.04  # KL to reference policy
    # Sampling during rollout (paper: temperature 1.0). We expose top_p / top_k / repetition
    # penalty so we can dial up diversity when the base model is overly confident (Qwen3 collapses
    # onto one answer with num_return_sequences=G at temp=1.0 → identical group → no advantage).
    train_temperature: float = 1.0
    train_top_p: float = 1.0
    train_top_k: int = 0            # 0 disables top-k
    train_repetition_penalty: float = 1.0
    eval_temperature: float = 0.0
    max_prompt_length: int = 4096  # paper
    max_response_length: int = 2048  # paper
    total_steps: int = 200  # paper Figure 7 shows ~200 steps
    save_every: int = 50
    seed: int = 42


@dataclass
class DataConfig:
    manager_data: Path = Path("data/processed/manager_train.jsonl")
    answer_data: Path = Path("data/processed/answer_train.jsonl")
    val_data: Path | None = Path("data/processed/answer_val.jsonl")
    max_examples: int | None = None
    top_k_context: int = 10  # for Manager: RAG-hits shown as old_memory
    top_k_per_speaker: int = 30  # for Answer: 30 per speaker => 60 total


@dataclass
class RewardConfig:
    """The reward for both agents is EM in the paper (Table 2 favours EM over J)."""

    reward_type: Literal["em", "f1", "judge"] = "em"
    judge_model: str = "gpt-4o-mini"


@dataclass
class AnswerBackendConfig:
    """Answer-Agent backend used *inside* Manager training as the frozen judge of the Manager's op."""

    name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    checkpoint: str | None = None  # optional path to an already RL-trained Answer Agent


@dataclass
class TrainerConfig:
    """Top-level config for a training run (Manager or Answer)."""

    stage: Literal["manager", "answer"] = "manager"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    answer_backend: AnswerBackendConfig = field(default_factory=AnswerBackendConfig)
    output_dir: Path = Path("outputs/checkpoints")
    run_name: str = "memory_r1_run"
    log_with: Literal["none", "wandb", "tensorboard"] = "none"


# --------------------------------------------------------------------------- loader


def load_trainer_config(path: str | Path) -> TrainerConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    def _dc(cls, data: dict | None):
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    return TrainerConfig(
        stage=raw.get("stage", "manager"),
        model=_dc(ModelConfig, raw.get("model")),
        optim=_dc(OptimConfig, raw.get("optim")),
        rl=_dc(RLConfig, raw.get("rl")),
        data=_dc(DataConfig, raw.get("data")),
        reward=_dc(RewardConfig, raw.get("reward")),
        answer_backend=_dc(AnswerBackendConfig, raw.get("answer_backend")),
        output_dir=Path(raw.get("output_dir", "outputs/checkpoints")),
        run_name=raw.get("run_name", "memory_r1_run"),
        log_with=raw.get("log_with", "none"),
    )
