from __future__ import annotations

import importlib.util
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


SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stability_selected_rag.py"
SPEC = importlib.util.spec_from_file_location("run_stability_selected_rag", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
selected_rag_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selected_rag_script)


class RunStabilitySelectedRagTests(unittest.TestCase):
    def test_selected_rag_reuses_existing_round_and_only_runs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sampling_root = temp / "source_sampling"
            output_root = temp / "selected_rag"

            for group_name in ("group_01", "group_02"):
                source_round_dir = sampling_root / group_name / "models" / "deepseek-v3-2" / "round_001"
                source_round_dir.mkdir(parents=True, exist_ok=True)
                (source_round_dir / "direct_answers.jsonl").write_text(
                    json.dumps(
                        {
                            "question": f"{group_name}-q1",
                            "ground_truth": "a1",
                            "record_type": "jsonl",
                            "tags": ["ragppi"],
                            "expected_answer_contains": [],
                            "expected_source_contains": [],
                            "expected_table_contains": [],
                            "expected_primary_ids": [],
                            "answer": "direct",
                            "status": "ok",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (sampling_root / group_name / "per_round_results.csv").write_text("round,seed\n1,42\n", encoding="utf-8")

            existing_round_dir = output_root / "sampling" / "group_01" / "models" / "deepseek-v3-2" / "round_001"
            existing_round_dir.mkdir(parents=True, exist_ok=True)
            (existing_round_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "rag": {"method": "RAG", "answer_relevancy": 0.8, "eval_mode": "local"},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(selected_rag_script.compare, "_build_rag_results", return_value=[{"question": "q1", "answer": "rag", "status": "ok"}]) as build_rag_results,
                mock.patch.object(selected_rag_script.compare, "_resolve_eval_mode", return_value=("local", None)),
                mock.patch.object(selected_rag_script.compare, "evaluate_local_results", return_value=selected_rag_script.pd.DataFrame([{"answer_relevancy": 0.9}])),
                mock.patch.object(selected_rag_script.compare, "compute_metrics", side_effect=lambda frame, method, extra=None: {"method": method, "answer_relevancy": 0.9, **(extra or {})}),
                mock.patch.object(selected_rag_script, "write_metric_plot", return_value=None),
            ):
                summary = selected_rag_script.run_stability_selected_rag(
                    source_sampling_root=sampling_root,
                    output_root=output_root,
                    benchmark_models=["DeepSeek-V3.2"],
                    sample_source_model="DeepSeek-V3.2",
                    group_start=1,
                    group_end=2,
                    eval_mode="local",
                    skip_existing=True,
                )

            self.assertEqual(build_rag_results.call_count, 1)
            self.assertEqual(summary["round_count"], 2)
            self.assertTrue((output_root / "models" / "deepseek-v3-2" / "model_summary.csv").exists())
            self.assertTrue((output_root / "per_round_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
