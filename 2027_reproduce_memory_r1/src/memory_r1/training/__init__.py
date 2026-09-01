"""RL trainers for Memory-R1.

The paper uses VERL (Sheng et al., 2025) for PPO/GRPO fine-tuning. Since VERL is heavier and less
portable, we implement the same training semantics on top of TRL (>= 0.12), which ships
``PPOTrainer`` and ``GRPOTrainer`` classes with the exact objective the paper describes.

Two training loops are exposed:

- :func:`memory_r1.training.entrypoints.train_memory_manager` — RL fine-tune the Manager. The
  reward comes from the frozen Answer Agent's EM against the gold answer, matching Section 3.1
  (Reward Design for Memory Manager) of the paper.
- :func:`memory_r1.training.entrypoints.train_answer_agent` — RL fine-tune the Answer Agent. The
  reward is EM(y_pred, y_gold) directly, per Section 3.2.
"""

from memory_r1.training.entrypoints import train_answer_agent, train_memory_manager
from memory_r1.training.reward_pipeline import (
    AnswerAgentRewardFn,
    ManagerRewardFn,
    build_answer_reward_fn,
    build_manager_reward_fn,
)

__all__ = [
    "train_memory_manager",
    "train_answer_agent",
    "ManagerRewardFn",
    "AnswerAgentRewardFn",
    "build_manager_reward_fn",
    "build_answer_reward_fn",
]
