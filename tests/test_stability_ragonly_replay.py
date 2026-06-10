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


SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stability_ragonly_replay.py"
SPEC = importlib.util.spec_from_file_location("run_stability_ragonly_replay", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
replay_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay_script)


class StabilityRagOnlyReplayTests(unittest.TestCase):
    def test_replay_reuses_direct_outputs_and_writes_group_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            main_dir = temp / "main"
            sampling_root = temp / "source_sampling"
            output_root = temp / "replay"
            main_dir.mkdir(parents=True, exist_ok=True)

            (main_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "benchmark_models": [
                            {"label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "slug": "deepseek-v3-2"}
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            round_dir = sampling_root / "group_01" / "models" / "deepseek-v3-2" / "round_001"
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "direct_answers.jsonl").write_text(
                json.dumps(
                    {
                        "question": "q1",
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
            (round_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "rag": {"method": "RAG", "answer_relevancy": 0.8},
                        "direct": {"method": "Direct", "answer_relevancy": 0.6, "direct_source": "model"},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(replay_script.compare, "_build_rag_results", return_value=[{"question": "q1", "answer": "rag", "status": "ok"}]),
                mock.patch.object(replay_script.compare, "_resolve_eval_mode", return_value=("local", None)),
                mock.patch.object(replay_script.compare, "evaluate_local_results", return_value=replay_script.pd.DataFrame([{"answer_relevancy": 0.9}])),
                mock.patch.object(replay_script.compare, "compute_metrics", side_effect=lambda frame, method, extra=None: {"method": method, "answer_relevancy": 0.9 if method == "RAG" else 0.6, **(extra or {})}),
                mock.patch.object(replay_script.compare, "_build_comparison_frame", return_value=replay_script.pd.DataFrame([{"method": "RAG", "answer_relevancy": 0.9}, {"method": "Direct", "answer_relevancy": 0.6}])),
                mock.patch.object(replay_script.compare, "_write_metric_plot", return_value=None),
                mock.patch.object(replay_script, "aggregate_formal_benchmark", return_value=None),
            ):
                summary = replay_script.run_stability_ragonly_replay(
                    main_dir=main_dir,
                    source_sampling_root=sampling_root,
                    output_root=output_root,
                    eval_mode="local",
                    aggregate=False,
                )

            copied_direct = output_root / "sampling" / "group_01" / "models" / "deepseek-v3-2" / "round_001" / "direct_answers.jsonl"
            self.assertTrue(copied_direct.exists())
            self.assertTrue((output_root / "sampling" / "group_01" / "per_round_results.csv").exists())
            self.assertEqual(summary["group_count"], 1)
            self.assertEqual(summary["model_count"], 1)


if __name__ == "__main__":
    unittest.main()
