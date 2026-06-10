from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCRIPT_PATH = REPO_ROOT / "scripts" / "filter_multimodel_report.py"
SPEC = importlib.util.spec_from_file_location("filter_multimodel_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
filter_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(filter_report)


class FilterMultimodelReportTests(unittest.TestCase):
    def test_filter_multimodel_report_keeps_only_requested_completed_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "models" / "qwen3").mkdir(parents=True)
            (input_dir / "models" / "minimax-m2").mkdir(parents=True)

            (input_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/fbtp_eval_fixed_120.jsonl",
                        "benchmark_models": [
                            {"label": "Qwen3", "model": "Qwen3", "slug": "qwen3"},
                            {"label": "MiniMax-M2", "model": "MiniMax-M2", "slug": "minimax-m2"},
                            {"label": "MiniMax-M2.7", "model": "MiniMax-M2.7", "slug": "minimax-m2-7"},
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            for slug in ("qwen3", "minimax-m2"):
                pd.DataFrame([{"model_label": slug, "method": "RAG", "answer_relevancy": 0.8}]).to_csv(
                    input_dir / "models" / slug / "model_summary.csv",
                    index=False,
                )
                pd.DataFrame([{"model_label": slug, "round_index": 1}]).to_csv(
                    input_dir / "models" / slug / "per_round_results.csv",
                    index=False,
                )
                pd.DataFrame([{"model_label": slug, "round_index": 1}]).to_csv(
                    input_dir / "models" / slug / "sampling_rounds.csv",
                    index=False,
                )
                (input_dir / "models" / slug / "summary.json").write_text(
                    json.dumps({"slug": slug}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            summary = filter_report.filter_multimodel_report(
                input_dir,
                output_dir,
                model_slugs=["qwen3", "minimax-m2"],
            )

            self.assertEqual(summary["model_count"], 2)
            filtered_config = json.loads((output_dir / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [spec["slug"] for spec in filtered_config["benchmark_models"]],
                ["qwen3", "minimax-m2"],
            )
            self.assertTrue((output_dir / "models" / "qwen3" / "summary.json").exists())
            self.assertTrue((output_dir / "models" / "minimax-m2" / "summary.json").exists())
            self.assertFalse((output_dir / "models" / "minimax-m2-7").exists())

            merged_summary = pd.read_csv(output_dir / "model_summary.csv")
            self.assertEqual(set(merged_summary["model_label"].tolist()), {"qwen3", "minimax-m2"})


if __name__ == "__main__":
    unittest.main()
