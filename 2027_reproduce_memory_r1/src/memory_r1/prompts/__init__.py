"""Prompt templates ported verbatim from paper Figures 9-12."""

from memory_r1.prompts.answer import (
    ANSWER_AGENT_SYSTEM,
    ANSWER_AGENT_SYSTEM_FOCUSED,
    ANSWER_AGENT_SYSTEM_PAPER,
    build_answer_prompt,
)
from memory_r1.prompts.extractor import EXTRACTOR_SYSTEM, build_extractor_prompt
from memory_r1.prompts.judge import JUDGE_SYSTEM, build_judge_prompt
from memory_r1.prompts.manager import MEMORY_MANAGER_SYSTEM, build_manager_prompt

__all__ = [
    "MEMORY_MANAGER_SYSTEM",
    "build_manager_prompt",
    "ANSWER_AGENT_SYSTEM",
    "ANSWER_AGENT_SYSTEM_FOCUSED",
    "ANSWER_AGENT_SYSTEM_PAPER",
    "build_answer_prompt",
    "JUDGE_SYSTEM",
    "build_judge_prompt",
    "EXTRACTOR_SYSTEM",
    "build_extractor_prompt",
]
