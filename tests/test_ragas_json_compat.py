from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import BaseModel, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.ragas_json_compat import (
    install_ragas_fenced_json_compat,
    uninstall_ragas_fenced_json_compat,
)


class _AnswerModel(BaseModel):
    answer: str


class RagasJsonCompatTests(unittest.TestCase):
    def tearDown(self) -> None:
        uninstall_ragas_fenced_json_compat()

    def test_fenced_json_is_accepted_after_patch(self) -> None:
        fenced = '```json\n{"answer":"ok"}\n```'
        uninstall_ragas_fenced_json_compat()

        with self.assertRaises(ValidationError):
            _AnswerModel.model_validate_json(fenced)

        install_ragas_fenced_json_compat()
        parsed = _AnswerModel.model_validate_json(fenced)

        self.assertEqual(parsed.answer, "ok")

    def test_patch_is_idempotent(self) -> None:
        install_ragas_fenced_json_compat()
        install_ragas_fenced_json_compat()
        parsed = _AnswerModel.model_validate_json('{"answer":"still-ok"}')
        self.assertEqual(parsed.answer, "still-ok")


if __name__ == "__main__":
    unittest.main()
