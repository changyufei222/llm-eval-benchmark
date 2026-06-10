from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "repair_ragas_answer_relevancy.py"
SPEC = importlib.util.spec_from_file_location("repair_ragas_answer_relevancy", SCRIPT_PATH)
repair_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repair_script)


class RepairAnswerRelevancyTests(unittest.TestCase):
    def test_merge_answer_relevancy_scores_only_fills_missing(self) -> None:
        existing = pd.DataFrame(
            [
                {"user_input": "q1", "answer_relevancy": None, "faithfulness": 0.8},
                {"user_input": "q2", "answer_relevancy": 0.4, "faithfulness": 0.7},
            ]
        )
        repaired = pd.DataFrame(
            [
                {"user_input": "q1", "answer_relevancy": 0.9},
                {"user_input": "q2", "answer_relevancy": 0.2},
            ]
        )

        merged = repair_script.merge_answer_relevancy_scores(existing, repaired)

        self.assertAlmostEqual(float(merged.loc[0, "answer_relevancy"]), 0.9, places=4)
        self.assertAlmostEqual(float(merged.loc[1, "answer_relevancy"]), 0.4, places=4)
        self.assertAlmostEqual(float(merged.loc[0, "faithfulness"]), 0.8, places=4)

    def test_replace_round_summary_updates_first_matching_method(self) -> None:
        frame = pd.DataFrame(
            [
                {"method": "RAG", "answer_relevancy": None, "faithfulness": 0.8, "round": 1},
                {"method": "Direct", "answer_relevancy": 0.1, "round": 1},
            ]
        )

        updated = repair_script.replace_round_summary(
            frame,
            "RAG",
            {"answer_relevancy": 0.77, "faithfulness": 0.88, "rows": 120},
        )

        rag_row = updated[updated["method"] == "RAG"].iloc[0]
        direct_row = updated[updated["method"] == "Direct"].iloc[0]
        self.assertAlmostEqual(float(rag_row["answer_relevancy"]), 0.77, places=4)
        self.assertAlmostEqual(float(rag_row["faithfulness"]), 0.88, places=4)
        self.assertAlmostEqual(float(direct_row["answer_relevancy"]), 0.1, places=4)


if __name__ == "__main__":
    unittest.main()
