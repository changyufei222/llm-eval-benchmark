from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from ragas.prompt.utils import extract_json


_ORIGINAL_MODEL_VALIDATE_JSON = None


def install_ragas_fenced_json_compat() -> None:
    global _ORIGINAL_MODEL_VALIDATE_JSON

    if _ORIGINAL_MODEL_VALIDATE_JSON is not None:
        return

    _ORIGINAL_MODEL_VALIDATE_JSON = BaseModel.__dict__["model_validate_json"].__func__

    def _patched_model_validate_json(cls, json_data: Any, *args: Any, **kwargs: Any):
        if isinstance(json_data, str):
            json_data = extract_json(json_data)
        return _ORIGINAL_MODEL_VALIDATE_JSON(cls, json_data, *args, **kwargs)

    BaseModel.model_validate_json = classmethod(_patched_model_validate_json)


def uninstall_ragas_fenced_json_compat() -> None:
    global _ORIGINAL_MODEL_VALIDATE_JSON

    if _ORIGINAL_MODEL_VALIDATE_JSON is None:
        return

    BaseModel.model_validate_json = classmethod(_ORIGINAL_MODEL_VALIDATE_JSON)
    _ORIGINAL_MODEL_VALIDATE_JSON = None
