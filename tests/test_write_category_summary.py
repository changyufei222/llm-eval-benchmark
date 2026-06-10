from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "write_category_summary.py"
SPEC = importlib.util.spec_from_file_location("write_category_summary", SCRIPT_PATH)
write_category_summary_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(write_category_summary_script)


class WriteCategorySummaryTests(unittest.TestCase):
    def test_write_category_summary_creates_expected_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_dir = root / "main"
            round_dir = main_dir / "models" / "deepseek-v3-2" / "round_001"
            round_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "benchmark_models": [
                            {
                                "label": "DeepSeek-V3.2",
                                "model": "DeepSeek-V3.2",
                                "slug": "deepseek-v3-2",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"method": "RAG", "category": "doc_design", "answer_relevancy": 0.8, "rows": 2},
                    {"method": "Direct", "category": "schema_tables", "answer_relevancy": 0.4, "rows": 3},
                ]
            ).to_csv(round_dir / "category_breakdown.csv", index=False)

            out_dir = root / "category_out"
            returned = write_category_summary_script.write_category_summary(main_dir, out_dir)

            self.assertEqual(returned, out_dir)
            self.assertTrue((out_dir / "category_summary.csv").exists())
            self.assertTrue((out_dir / "category_summary.md").exists())
            self.assertTrue((out_dir / "summary.md").exists())

            summary = pd.read_csv(out_dir / "category_summary.csv")
            self.assertEqual(set(summary["category_bucket"]), {"doc/design", "schema_tables"})


if __name__ == "__main__":
    unittest.main()
