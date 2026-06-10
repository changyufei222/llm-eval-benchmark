from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sys
from importlib import import_module


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class CompareOutputTests(unittest.TestCase):
    def test_write_metric_plot_creates_png(self) -> None:
        plotting = import_module("metrics.plotting")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparison_summary.png"
            frame = pd.DataFrame(
                [
                    {
                        "method": "RAG",
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                        "context_precision": 0.7,
                    },
                    {
                        "method": "Direct",
                        "faithfulness": None,
                        "answer_relevancy": 0.6,
                        "context_precision": None,
                    },
                ]
            )

            plotting.write_metric_plot(frame, output_path, title="Eval Comparison")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
