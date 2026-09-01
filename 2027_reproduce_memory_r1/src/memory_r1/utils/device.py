"""Hardware auto-detection.

The same code base needs to run on:
  * Apple silicon (MPS) locally,
  * 1× H100 (CUDA) in the cloud,
  * CPU as a last-resort fallback.

``resolve_device`` picks the best backend at runtime; ``resolve_dtype`` maps a config-string
dtype to a ``torch.dtype`` with per-device sanity (bf16 is painfully slow on CPU, so we
silently upcast to fp32 there).
"""

from __future__ import annotations

import os
from typing import Literal

import torch

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

DeviceStr = Literal["cuda", "mps", "cpu"]


def resolve_device(preferred: str | None = None) -> str:
    """Return the best available torch device.

    Order: explicit ``preferred`` (if available) > env ``MEMORY_R1_DEVICE`` > CUDA > MPS > CPU.
    """

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    env = os.environ.get("MEMORY_R1_DEVICE")
    if env:
        candidates.append(env)
    candidates.extend(["cuda", "mps", "cpu"])

    for cand in candidates:
        if cand == "cuda" and torch.cuda.is_available():
            return "cuda"
        if cand == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if cand == "cpu":
            return "cpu"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Map a config dtype string to ``torch.dtype``, adjusting for the target device.

    * On CPU, ``bfloat16`` is technically supported but ~10× slower than fp32 for most ops —
      silently upcast so users don't accidentally shoot themselves in the foot.
    * On MPS, ``bfloat16`` is supported for PyTorch >= 2.3; older builds fall back to fp16.
    """

    if requested not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {requested!r}; expected one of {list(DTYPE_MAP)}")
    dtype = DTYPE_MAP[requested]

    if device == "cpu" and dtype == torch.bfloat16:
        return torch.float32
    if device == "mps" and dtype == torch.bfloat16:
        # PyTorch >= 2.3 supports bf16 on MPS. If not, fall back to fp16.
        if not _mps_supports_bf16():
            return torch.float16
    return dtype


def _mps_supports_bf16() -> bool:
    try:
        _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except Exception:
        return False
