from __future__ import annotations

import re
from typing import Sequence


CONTROLLED_MAIN_MODELS: tuple[str, ...] = (
    "DeepSeek-V3.2",
    "GLM-5",
    "Kimi-K2.5",
    "DeepSeek-R1",
    "Qwen3-235B-A22B-Instruct-2507",
    "MiniMax-M2",
    "MiniMax-M2.7",
    "ERNIE-4.5-Turbo-128K",
)

STABILITY_ANCHOR_MODELS: tuple[str, ...] = CONTROLLED_MAIN_MODELS[:4]
APPENDIX_NATIVE_MODELS: tuple[str, ...] = CONTROLLED_MAIN_MODELS
MISSING_MAIN_MODELS: tuple[str, ...] = CONTROLLED_MAIN_MODELS[4:]
MISSING_APPENDIX_MODELS: tuple[str, ...] = (
    "DeepSeek-V3.2",
    "Qwen3-235B-A22B-Instruct-2507",
    "MiniMax-M2",
    "MiniMax-M2.7",
    "ERNIE-4.5-Turbo-128K",
)
TOPK_CANDIDATE_VALUES: tuple[int, ...] = (32, 64, 128, 256)
FINAL_CONTEXT_TOP_K: int = 5

_THINKING_DISABLED_PREFIXES: tuple[str, ...] = (
    "deepseekv3",
    "glm",
    "kimi",
    "qwen",
    "minimax",
    "ernie",
)


def normalized_model_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(model).lower())


def controlled_thinking_mode(model: str) -> str | None:
    normalized = normalized_model_name(model)
    if normalized.startswith("deepseekr1"):
        return None
    if any(normalized.startswith(prefix) for prefix in _THINKING_DISABLED_PREFIXES):
        return "disabled"
    return None


def benchmark_model_flags(models: Sequence[str], *, indent: str = "  --benchmark-model ") -> str:
    lines = [f"{indent}{model}" for model in models]
    return " \\\n".join(lines)
