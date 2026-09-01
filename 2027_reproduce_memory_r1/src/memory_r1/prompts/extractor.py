"""Fact-extraction prompt.

The paper (Appendix B.2, Algorithm 3 line ``f_i <- LLMExtract(t_i)``) uses GPT-4o-mini to extract
key facts from each dialogue turn. The exact prompt is not shown in the paper, but the extraction
follows the Mem0 setup (Chhikara et al., 2025) which is what Memory-R1's prompts are adapted from.
We stay faithful to that convention: standalone atomic facts, third-person, no pronouns.
"""

from __future__ import annotations

EXTRACTOR_SYSTEM = """\
You extract atomic factual statements from a dialogue turn.

Given a single dialogue turn "<speaker>: <utterance>", output a JSON object with a top-level "facts"
key mapped to a list of standalone factual statements.

Rules:
- Each fact must be an atomic, self-contained sentence.
- Refer to the speaker by their name (not "I", "me", "my"). E.g. instead of "I love pizza" output
  "Andrew loves pizza".
- Only include facts that carry information about the speaker, another person, or the world.
  Discard greetings, filler, backchannels, and questions.
- If the turn has no extractable facts, return {"facts": []}.
- Keep facts short (< 20 words).
- Do not add explanations. Output JSON only.
"""


def build_extractor_prompt(speaker: str, utterance: str, timestamp: str | None = None) -> str:
    header = f"Speaker: {speaker}"
    if timestamp:
        header += f"\nTimestamp: {timestamp}"
    return f"{header}\nUtterance: {utterance}\n\nReturn the JSON."
