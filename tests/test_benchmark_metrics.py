from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.metrics import build_category_breakdown, estimate_row_cost, evaluate_local_results


class BenchmarkMetricTests(unittest.TestCase):
    def test_estimate_row_cost_uses_usage_when_available(self) -> None:
        row = {
            "prompt_tokens": 1000,
            "completion_tokens": 250,
        }
        cost = estimate_row_cost(row, input_cost_per_million=2.0, output_cost_per_million=8.0)
        self.assertAlmostEqual(cost, 0.004, places=6)

    def test_category_breakdown_groups_by_primary_tag(self) -> None:
        frame = pd.DataFrame(
            [
                {"method": "RAG", "category": "ragppi", "answer_token_f1": 0.4, "latency_ms": 100},
                {"method": "RAG", "category": "ragppi", "answer_token_f1": 0.6, "latency_ms": 120},
                {"method": "Direct", "category": "ragppi", "answer_token_f1": 0.2, "latency_ms": 300},
                {"method": "RAG", "category": "docx", "answer_token_f1": 0.8, "latency_ms": 80},
            ]
        )
        breakdown = build_category_breakdown(frame)
        self.assertEqual(set(breakdown["category"]), {"ragppi", "docx"})
        self.assertEqual(set(breakdown["method"]), {"RAG", "Direct"})
        ragppi = breakdown[(breakdown["category"] == "ragppi") & (breakdown["method"] == "RAG")].iloc[0]
        self.assertAlmostEqual(ragppi["answer_token_f1"], 0.5, places=4)

    def test_local_metrics_include_table_and_id_hit_rates(self) -> None:
        results = [
            {
                "question": "What is the affinity for INT-00001?",
                "category": "affinity",
                "answer": "Affinity is 6.2 nM.",
                "ground_truth": "Affinity is 6.2 nM.",
                "source_markers": ["interaction_cards.jsonl"],
                "table_markers": ["interactions", "affinity_data"],
                "id_markers": ["INT-00001", "TBP-00007"],
                "expected_answer_contains": ["6.2"],
                "expected_source_contains": ["interaction_cards.jsonl"],
                "expected_table_contains": ["interactions", "affinity_data"],
                "expected_primary_ids": ["INT-00001"],
                "context_count": 1,
                "context_scores": [0.9],
                "latency_ms": 12.0,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "estimated_cost_usd": 0.0,
            }
        ]
        frame = evaluate_local_results(results)
        row = frame.iloc[0]
        self.assertAlmostEqual(row["table_hit_rate"], 1.0, places=4)
        self.assertAlmostEqual(row["id_hit_rate"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
