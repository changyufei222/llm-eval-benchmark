from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


fake_pgvector_store = types.ModuleType("ragkb.storage.pgvector_store")
fake_pgvector_store.keyword_search = lambda *args, **kwargs: []
fake_pgvector_store.search = lambda *args, **kwargs: []
sys.modules.setdefault("ragkb.storage.pgvector_store", fake_pgvector_store)


from pipelines import compare
from ragkb.config import Settings


class CompareSamplingTests(unittest.TestCase):
    def test_parse_benchmark_model_specs_supports_plain_and_labeled_names(self) -> None:
        specs = compare._parse_benchmark_model_specs(
            ["baseline=DeepSeek-V3.2", "GLM-5", "reasoner=DeepSeek-R1"],
            default_model="DeepSeek-V3.2",
        )

        self.assertEqual(
            specs,
            [
                {
                    "label": "baseline",
                    "model": "DeepSeek-V3.2",
                    "slug": "baseline",
                },
                {
                    "label": "GLM-5",
                    "model": "GLM-5",
                    "slug": "glm-5",
                },
                {
                    "label": "reasoner",
                    "model": "DeepSeek-R1",
                    "slug": "reasoner",
                },
            ],
        )

    def test_sample_rows_with_replacement_preserves_requested_size(self) -> None:
        rows = [{"question": f"q{i}"} for i in range(3)]

        sampled = compare._sample_rows(rows, sample_size=5, with_replacement=True, seed=7)

        self.assertEqual(len(sampled), 5)
        self.assertTrue(all("question" in row for row in sampled))

    def test_sample_rows_without_replacement_rejects_oversized_sample(self) -> None:
        rows = [{"question": f"q{i}"} for i in range(3)]

        with self.assertRaises(ValueError):
            compare._sample_rows(rows, sample_size=4, with_replacement=False, seed=7)

    def test_settings_with_candidate_top_k_updates_reranker_depth(self) -> None:
        settings = Settings(reranker_top_n=8)

        updated = compare._settings_with_candidate_top_k(settings, candidate_top_k=64)

        self.assertEqual(updated.reranker_top_n, 64)

    def test_run_compare_multimodel_sampling_records_candidate_top_k_in_experiment_config(self) -> None:
        rows = [{"question": f"q{i}", "ground_truth": "a"} for i in range(4)]

        def fake_run_compare_sampling(
            sampled_rows,
            out_dir,
            *,
            model_label,
            model_name,
            settings,
            **kwargs,
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            round_records = [
                {
                    "model_label": model_label,
                    "model": model_name,
                    "method": "RAG",
                    "round": 1,
                    "seed": 42,
                    "sample_size": len(sampled_rows),
                    "unique_questions": len(sampled_rows),
                    "answer_relevancy": 0.8,
                },
                {
                    "model_label": model_label,
                    "model": model_name,
                    "method": "Direct",
                    "round": 1,
                    "seed": 42,
                    "sample_size": len(sampled_rows),
                    "unique_questions": len(sampled_rows),
                    "answer_relevancy": 0.6,
                },
            ]
            (out_dir / "per_round_results.csv").write_text(
                "model_label,model,method,round,seed,sample_size,unique_questions,answer_relevancy\n"
                f"{model_label},{model_name},RAG,1,42,{len(sampled_rows)},{len(sampled_rows)},0.8\n"
                f"{model_label},{model_name},Direct,1,42,{len(sampled_rows)},{len(sampled_rows)},0.6\n",
                encoding="utf-8",
            )
            (out_dir / "model_summary.csv").write_text(
                "model_label,model,method,answer_relevancy_mean\n"
                f"{model_label},{model_name},RAG,0.8\n"
                f"{model_label},{model_name},Direct,0.6\n",
                encoding="utf-8",
            )
            self.assertEqual(settings.reranker_top_n, 128)
            return {
                "model_label": model_label,
                "model": model_name,
                "rounds": kwargs["rounds"],
                "sample_size": kwargs["sample_size"],
                "per_round_results_path": str(out_dir / "per_round_results.csv"),
                "model_summary_path": str(out_dir / "model_summary.csv"),
                "round_records": round_records,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"
            with mock.patch.object(compare, "_run_compare_sampling", side_effect=fake_run_compare_sampling):
                compare._run_compare_multimodel_sampling(
                    rows,
                    out_dir,
                    settings=Settings(reranker_top_n=8),
                    top_k=5,
                    candidate_top_k=128,
                    eval_mode_requested="local",
                    fail_on_fallback=False,
                    data_path_label="data/fixed.jsonl",
                    rounds=2,
                    sample_size=3,
                    with_replacement=True,
                    seed=42,
                    benchmark_models=[
                        {"label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "slug": "deepseek-v3-2"},
                    ],
                )

            config = json.loads((out_dir / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["top_k"], 5)
            self.assertEqual(config["candidate_top_k"], 128)

    def test_run_compare_sampling_writes_round_and_summary_outputs(self) -> None:
        rows = [{"question": f"q{i}", "ground_truth": "a"} for i in range(4)]

        def fake_run_compare_once(sampled_rows, out_dir, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "data_path": kwargs["data_path_label"],
                "samples": len(sampled_rows),
                "eval_mode": "local",
                "rag": {
                    "method": "RAG",
                    "answer_relevancy": 0.8,
                    "source_hit_rate": 0.9,
                    "latency_ms": 100.0,
                    "estimated_cost_usd": 0.01,
                },
                "direct": {
                    "method": "Direct",
                    "answer_relevancy": 0.6,
                    "source_hit_rate": 0.3,
                    "latency_ms": 80.0,
                    "estimated_cost_usd": 0.02,
                },
            }
            (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            return payload

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"
            with mock.patch.object(compare, "_run_compare_once", side_effect=fake_run_compare_once):
                summary = compare._run_compare_sampling(
                    rows,
                    out_dir,
                    settings=mock.Mock(),
                    top_k=5,
                    eval_mode_requested="local",
                    direct_model="gpt-test",
                    fail_on_fallback=False,
                    data_path_label="data/fixed.jsonl",
                    rounds=3,
                    sample_size=2,
                    with_replacement=True,
                    seed=11,
                    model_label="DeepSeek-V3.2",
                    model_name="DeepSeek-V3.2",
                )

            rounds_csv = out_dir / "sampling_rounds.csv"
            per_round_csv = out_dir / "per_round_results.csv"
            model_summary_csv = out_dir / "model_summary.csv"
            leaderboard_csv = out_dir / "leaderboard.csv"
            summary_json = out_dir / "summary.json"
            comparison_csv = out_dir / "comparison.csv"

            self.assertTrue(rounds_csv.exists())
            self.assertTrue(per_round_csv.exists())
            self.assertTrue(model_summary_csv.exists())
            self.assertTrue(leaderboard_csv.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(comparison_csv.exists())
            self.assertEqual(summary["rounds"], 3)
            self.assertEqual(summary["sample_size"], 2)
            self.assertEqual(summary["model_label"], "DeepSeek-V3.2")

    def test_run_compare_multimodel_sampling_writes_experiment_outputs(self) -> None:
        rows = [{"question": f"q{i}", "ground_truth": "a"} for i in range(4)]

        def fake_run_compare_sampling(
            sampled_rows,
            out_dir,
            *,
            model_label,
            model_name,
            **kwargs,
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            round_records = [
                {
                    "model_label": model_label,
                    "model": model_name,
                    "method": "RAG",
                    "round": 1,
                    "seed": 42,
                    "sample_size": len(sampled_rows),
                    "unique_questions": len(sampled_rows),
                    "answer_relevancy": 0.8,
                },
                {
                    "model_label": model_label,
                    "model": model_name,
                    "method": "Direct",
                    "round": 1,
                    "seed": 42,
                    "sample_size": len(sampled_rows),
                    "unique_questions": len(sampled_rows),
                    "answer_relevancy": 0.6,
                },
            ]
            (out_dir / "per_round_results.csv").write_text(
                "model_label,model,method,round,seed,sample_size,unique_questions,answer_relevancy\n"
                f"{model_label},{model_name},RAG,1,42,{len(sampled_rows)},{len(sampled_rows)},0.8\n"
                f"{model_label},{model_name},Direct,1,42,{len(sampled_rows)},{len(sampled_rows)},0.6\n",
                encoding="utf-8",
            )
            (out_dir / "model_summary.csv").write_text(
                "model_label,model,method,answer_relevancy_mean\n"
                f"{model_label},{model_name},RAG,0.8\n"
                f"{model_label},{model_name},Direct,0.6\n",
                encoding="utf-8",
            )
            return {
                "model_label": model_label,
                "model": model_name,
                "rounds": kwargs["rounds"],
                "sample_size": kwargs["sample_size"],
                "per_round_results_path": str(out_dir / "per_round_results.csv"),
                "model_summary_path": str(out_dir / "model_summary.csv"),
                "round_records": round_records,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"
            with mock.patch.object(compare, "_run_compare_sampling", side_effect=fake_run_compare_sampling):
                summary = compare._run_compare_multimodel_sampling(
                    rows,
                    out_dir,
                    settings=mock.Mock(),
                    top_k=5,
                    eval_mode_requested="local",
                    fail_on_fallback=False,
                    data_path_label="data/fixed.jsonl",
                    rounds=2,
                    sample_size=3,
                    with_replacement=True,
                    seed=42,
                    benchmark_models=[
                        {"label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "slug": "deepseek-v3-2"},
                        {"label": "GLM-5", "model": "GLM-5", "slug": "glm-5"},
                    ],
                )

            self.assertTrue((out_dir / "experiment_config.json").exists())
            self.assertTrue((out_dir / "per_round_results.csv").exists())
            self.assertTrue((out_dir / "model_summary.csv").exists())
            self.assertTrue((out_dir / "leaderboard.csv").exists())
            self.assertEqual(summary["model_count"], 2)
            self.assertEqual(summary["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
