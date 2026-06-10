from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines import compare


class CompareSubprocessTests(unittest.TestCase):
    def test_run_ragas_subprocess_forwards_llm_model_to_child_env(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None, env=None):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["env"] = env
            return types.SimpleNamespace(stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            (out_dir / "direct_ragas_scores.csv").write_text("answer_relevancy\n0.9\n", encoding="utf-8")
            (out_dir / "direct_ragas_summary.json").write_text('{"answer_relevancy": 0.9}', encoding="utf-8")

            with (
                mock.patch("pipelines.compare.subprocess.run", side_effect=fake_run),
                mock.patch("pipelines.compare.pd.read_csv", return_value=pd.DataFrame([{"answer_relevancy": 0.9}])),
                mock.patch("pipelines.compare._load_json", return_value={"answer_relevancy": 0.9}),
            ):
                compare._run_ragas_subprocess(
                    answers_path=out_dir / "direct_answers.jsonl",
                    out_dir=out_dir,
                    metric_set="direct",
                    model="GLM-5",
                )

        self.assertEqual(captured["env"]["LLM_MODEL"], "GLM-5")


if __name__ == "__main__":
    unittest.main()
