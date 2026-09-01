"""High-level entrypoints called by the CLI.

- :func:`train_memory_manager` — trains the Memory Manager with PPO or GRPO. Wires up a *frozen*
  Answer Agent that scores each rollout (reward = EM against the QA linked to that turn).
- :func:`train_answer_agent` — trains the Answer Agent with PPO or GRPO. Reward is direct EM.
"""

from __future__ import annotations

from pathlib import Path

from memory_r1.agents.answer_agent import AnswerAgent
from memory_r1.agents.llm_backend import HFBackend
from memory_r1.memory.retrieval import DenseRetriever
from memory_r1.training.config import TrainerConfig, load_trainer_config
from memory_r1.training.reward_pipeline import (
    build_answer_reward_fn,
    build_manager_reward_fn,
)
from memory_r1.utils import logger


def train_memory_manager(config_path: str | Path) -> None:
    cfg: TrainerConfig = load_trainer_config(config_path)
    logger.info("🚀 Memory Manager RL fine-tuning")
    logger.info("⚙️  algorithm={} backbone={}", cfg.rl.algorithm.upper(), cfg.model.name_or_path)

    # Frozen Answer Agent used as the reward model.
    logger.info("📥 loading frozen Answer Agent from: {}", cfg.answer_backend.name_or_path)
    backend = HFBackend(
        model_name_or_path=cfg.answer_backend.checkpoint or cfg.answer_backend.name_or_path,
        dtype=cfg.answer_backend.dtype,
    )
    answer = AnswerAgent(backend=backend, temperature=0.0)
    retriever = DenseRetriever()
    reward_fn = build_manager_reward_fn(
        answer_agent=answer,
        retriever=retriever,
        top_k_per_speaker=cfg.data.top_k_per_speaker,
    )

    from memory_r1.training.grpo_manager import load_manager_examples

    examples = load_manager_examples(cfg.data.manager_data, max_examples=cfg.data.max_examples)
    logger.info("📚 loaded {} manager examples", len(examples))
    if not examples:
        raise RuntimeError(
            f"No Memory Manager training examples with linked QA at {cfg.data.manager_data}. "
            "Run `memory-r1 build-manager-data` first."
        )

    if cfg.rl.algorithm == "grpo":
        from memory_r1.training.grpo_manager import GRPOManagerTrainer

        trainer = GRPOManagerTrainer(cfg=cfg, reward_fn=reward_fn, examples=examples)
    elif cfg.rl.algorithm == "ppo":
        from memory_r1.training.ppo_manager import PPOManagerTrainer

        trainer = PPOManagerTrainer(cfg=cfg, reward_fn=reward_fn, examples=examples)
    else:
        raise ValueError(cfg.rl.algorithm)

    trainer.train()


def train_answer_agent(config_path: str | Path) -> None:
    cfg: TrainerConfig = load_trainer_config(config_path)
    logger.info("🚀 Answer Agent RL fine-tuning")
    logger.info("⚙️  algorithm={} backbone={}", cfg.rl.algorithm.upper(), cfg.model.name_or_path)

    from memory_r1.training.grpo_answer import load_answer_examples

    examples = load_answer_examples(cfg.data.answer_data, max_examples=cfg.data.max_examples)
    logger.info("📚 loaded {} answer examples", len(examples))
    if not examples:
        raise RuntimeError(
            f"No Answer Agent training examples at {cfg.data.answer_data}. "
            "Run `memory-r1 build-answer-data` first."
        )

    reward_fn = build_answer_reward_fn(reward_type=cfg.reward.reward_type)

    if cfg.rl.algorithm == "grpo":
        from memory_r1.training.grpo_answer import GRPOAnswerTrainer

        trainer = GRPOAnswerTrainer(cfg=cfg, reward_fn=reward_fn, examples=examples)
    elif cfg.rl.algorithm == "ppo":
        from memory_r1.training.ppo_answer import PPOAnswerTrainer

        trainer = PPOAnswerTrainer(cfg=cfg, reward_fn=reward_fn, examples=examples)
    else:
        raise ValueError(cfg.rl.algorithm)

    trainer.train()
