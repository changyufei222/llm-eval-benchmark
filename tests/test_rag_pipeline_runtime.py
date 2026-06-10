from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines import rag_pipeline


class RagPipelineRuntimeTests(unittest.TestCase):
    def test_main_passes_llm_and_embeddings_to_ragas(self) -> None:
        captured: dict[str, object] = {}

        def fake_evaluate(dataset, metrics, llm=None, embeddings=None):
            captured["llm"] = llm
            captured["embeddings"] = embeddings
            return mock.Mock(to_pandas=lambda: pd.DataFrame([{"faithfulness": 1.0, "answer_relevancy": 1.0}]))

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            data_path = tmp / "eval.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "question": "What is reported about KLK7 and CDSN?",
                        "ground_truth": "KLK7 cleaves CDSN.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = tmp / "reports"

            with (
                mock.patch.object(rag_pipeline, "retrieve", return_value=[{"content": "context"}]),
                mock.patch.object(rag_pipeline, "build_answer", return_value="answer"),
                mock.patch.object(rag_pipeline, "evaluate", side_effect=fake_evaluate),
                mock.patch.object(rag_pipeline.plt, "savefig"),
            ):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "rag_pipeline",
                        "--data-path",
                        str(data_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                ):
                    rag_pipeline.main()

        self.assertIsNotNone(captured.get("llm"))
        self.assertIsNotNone(captured.get("embeddings"))


if __name__ == "__main__":
    unittest.main()
