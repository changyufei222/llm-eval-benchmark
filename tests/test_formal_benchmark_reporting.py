from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.formal_benchmark import (  # type: ignore[attr-defined]
    combine_main_category_breakdowns,
    normalize_category_bucket,
    pivot_method_metrics,
    summarize_benchmark_distribution,
    summarize_rag_direct_uplift,
)


class FormalBenchmarkReportingTests(unittest.TestCase):
    def test_summarize_benchmark_distribution_adds_ci_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "round": 1, "seed": 101, "answer_relevancy": 0.4},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "round": 2, "seed": 102, "answer_relevancy": 0.6},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "round": 3, "seed": 103, "answer_relevancy": 0.8},
            ]
        )

        summary = summarize_benchmark_distribution(
            frame,
            group_cols=["model_label", "model", "method"],
            exclude_numeric_cols={"round", "seed"},
        )

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["rows"], 3)
        self.assertAlmostEqual(row["answer_relevancy_mean"], 0.6, places=4)
        self.assertAlmostEqual(row["answer_relevancy_std"], 0.2, places=4)
        self.assertAlmostEqual(row["answer_relevancy_p50"], 0.6, places=4)
        self.assertIn("answer_relevancy_ci95_low", summary.columns)
        self.assertIn("answer_relevancy_ci95_high", summary.columns)
        expected_half_width = 1.96 * 0.2 / math.sqrt(3)
        self.assertAlmostEqual(row["answer_relevancy_ci95_low"], 0.6 - expected_half_width, places=4)
        self.assertAlmostEqual(row["answer_relevancy_ci95_high"], 0.6 + expected_half_width, places=4)

    def test_summarize_rag_direct_uplift_uses_round_aligned_deltas(self) -> None:
        frame = pd.DataFrame(
            [
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "round": 1, "seed": 201, "answer_relevancy": 0.8},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "Direct", "round": 1, "seed": 201, "answer_relevancy": 0.5},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "round": 2, "seed": 202, "answer_relevancy": 0.7},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "Direct", "round": 2, "seed": 202, "answer_relevancy": 0.6},
            ]
        )

        summary = summarize_rag_direct_uplift(frame)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["comparison"], "RAG - Direct")
        self.assertEqual(row["rows"], 2)
        self.assertAlmostEqual(row["answer_relevancy_mean"], 0.2, places=4)
        self.assertAlmostEqual(row["answer_relevancy_p50"], 0.2, places=4)
        self.assertIn("answer_relevancy_ci95_low", summary.columns)
        self.assertIn("answer_relevancy_ci95_high", summary.columns)

    def test_summarize_rag_direct_uplift_ignores_identity_numeric_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "global_round": 1, "round": 1, "seed": 301, "answer_relevancy": 0.9},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "Direct", "global_round": 1, "round": 1, "seed": 301, "answer_relevancy": 0.4},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "RAG", "global_round": 2, "round": 1, "seed": 302, "answer_relevancy": 0.7},
                {"model_label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "method": "Direct", "global_round": 2, "round": 1, "seed": 302, "answer_relevancy": 0.5},
            ]
        )

        summary = summarize_rag_direct_uplift(frame)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertAlmostEqual(row["answer_relevancy_mean"], 0.35, places=4)
        self.assertNotIn("global_round_mean", summary.columns)

    def test_normalize_category_bucket_maps_presentation_groups(self) -> None:
        self.assertEqual(normalize_category_bucket("ragppi"), "ragppi")
        self.assertEqual(normalize_category_bucket("doc_design"), "doc/design")
        self.assertEqual(normalize_category_bucket("architecture"), "doc/design")
        self.assertEqual(normalize_category_bucket("schema_tables"), "schema_tables")
        self.assertEqual(normalize_category_bucket("protein_profile"), "schema_tables")

    def test_combine_main_category_breakdowns_rolls_up_weighted_category_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            main_dir = Path(temp_dir) / "main_fixed120"
            model_dir = main_dir / "models" / "deepseek-v3-2" / "round_001"
            model_dir.mkdir(parents=True, exist_ok=True)
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
                    {"method": "RAG", "category": "doc_design", "answer_relevancy": 0.4, "rows": 2},
                    {"method": "RAG", "category": "architecture", "answer_relevancy": 0.8, "rows": 1},
                    {"method": "RAG", "category": "protein_profile", "answer_relevancy": 0.9, "rows": 3},
                    {"method": "Direct", "category": "ragppi", "answer_relevancy": 0.5, "rows": 4},
                ]
            ).to_csv(model_dir / "category_breakdown.csv", index=False)

            summary = combine_main_category_breakdowns(main_dir)

        doc_row = summary[(summary["method"] == "RAG") & (summary["category_bucket"] == "doc/design")].iloc[0]
        schema_row = summary[(summary["method"] == "RAG") & (summary["category_bucket"] == "schema_tables")].iloc[0]
        ragppi_row = summary[(summary["method"] == "Direct") & (summary["category_bucket"] == "ragppi")].iloc[0]

        self.assertEqual(doc_row["model_label"], "DeepSeek-V3.2")
        self.assertEqual(doc_row["rows"], 3)
        self.assertAlmostEqual(doc_row["answer_relevancy"], 0.5333, places=4)
        self.assertEqual(schema_row["rows"], 3)
        self.assertAlmostEqual(schema_row["answer_relevancy"], 0.9, places=4)
        self.assertEqual(ragppi_row["rows"], 4)

    def test_pivot_method_metrics_keeps_expected_columns_when_rag_metric_is_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_label": "MiniMax-M2",
                    "category_bucket": "doc/design",
                    "method": "Direct",
                    "answer_relevancy": 0.0,
                    "latency_ms": 2911.747,
                    "rows": 20,
                },
                {
                    "model_label": "MiniMax-M2",
                    "category_bucket": "doc/design",
                    "method": "RAG",
                    "faithfulness": 0.8222,
                    "latency_ms": 177833.576,
                    "rows": 20,
                },
            ]
        )

        summary = pivot_method_metrics(
            frame,
            index_cols=["model_label", "category_bucket"],
            value_cols=["answer_relevancy", "faithfulness", "latency_ms", "rows"],
        )

        self.assertEqual(len(summary), 1)
        self.assertIn("answer_relevancy_direct", summary.columns)
        self.assertIn("answer_relevancy_rag", summary.columns)
        self.assertIn("faithfulness_direct", summary.columns)
        self.assertIn("faithfulness_rag", summary.columns)
        row = summary.iloc[0]
        self.assertAlmostEqual(row["answer_relevancy_direct"], 0.0, places=4)
        self.assertTrue(pd.isna(row["answer_relevancy_rag"]))
        self.assertTrue(pd.isna(row["faithfulness_direct"]))
        self.assertAlmostEqual(row["faithfulness_rag"], 0.8222, places=4)

    def test_pivot_method_metrics_does_not_create_cartesian_product_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_label": "DeepSeek-V3.2",
                    "model": "DeepSeek-V3.2",
                    "method": "Direct",
                    "answer_relevancy": 0.6,
                },
                {
                    "model_label": "DeepSeek-V3.2",
                    "model": "DeepSeek-V3.2",
                    "method": "RAG",
                    "answer_relevancy": 0.8,
                },
                {
                    "model_label": "GLM-5",
                    "model": "GLM-5",
                    "method": "Direct",
                    "answer_relevancy": 0.2,
                },
                {
                    "model_label": "GLM-5",
                    "model": "GLM-5",
                    "method": "RAG",
                    "answer_relevancy": 0.7,
                },
            ]
        )

        summary = pivot_method_metrics(
            frame,
            index_cols=["model_label", "model"],
            value_cols=["answer_relevancy"],
        )

        self.assertEqual(len(summary), 2)
        self.assertEqual(
            set(tuple(row) for row in summary[["model_label", "model"]].itertuples(index=False, name=None)),
            {
                ("DeepSeek-V3.2", "DeepSeek-V3.2"),
                ("GLM-5", "GLM-5"),
            },
        )


if __name__ == "__main__":
    unittest.main()
