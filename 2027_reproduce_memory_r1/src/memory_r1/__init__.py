"""Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning.

Reproduction of Yan et al., arXiv:2508.19828 (2026-01-14 v5).

The package is organized into:

- ``memory_r1.data``: dataset loaders (LoCoMo, MSC, LongMemEval) and data-construction pipelines
  (Algorithms 1 and 2 in the paper).
- ``memory_r1.memory``: the memory-bank data structure, ADD/UPDATE/DELETE/NOOP operations, and RAG
  retrieval on top of sentence-transformer embeddings.
- ``memory_r1.agents``: the fact extractor, Memory Manager, and Answer Agent (with Memory
  Distillation).
- ``memory_r1.prompts``: verbatim prompts from paper Figures 9-12.
- ``memory_r1.training``: PPO and GRPO trainers for both agents, together with the outcome-driven
  reward function.
- ``memory_r1.eval``: F1, BLEU-1, LLM-as-a-Judge metrics and the end-to-end evaluation pipeline.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
