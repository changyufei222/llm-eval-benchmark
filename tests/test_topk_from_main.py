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


SCRIPT_PATH = REPO_ROOT / "scripts" / "run_topk_from_main.py"
SPEC = importlib.util.spec_from_file_location("run_topk_from_main", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
topk_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topk_script)


class TopkFromMainTests(unittest.TestCase):
    def test_run_topk_from_main_reuses_direct_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            data_path = temp / "eval.jsonl"
            data_path.write_text(json.dumps({"question": "q1", "ground_truth": "a1"}) + "\n", encoding="utf-8")

            main_dir = temp / "main"
            round_dir = main_dir / "models" / "deepseek-v3-2" / "round_001"
            round_dir.mkdir(parents=True, exist_ok=True)
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
            (round_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "rag": {"method": "RAG", "answer_relevancy": 0.8},
                        "direct": {"method": "Direct", "answer_relevancy": 0.6},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (round_dir / "direct_answers.jsonl").write_text(
                json.dumps({"question": "q1", "answer": "direct"}) + "\n",
                encoding="utf-8",
            )

            output_dir = temp / "topk"

            with (
                mock.patch.object(topk_script.compare, "_build_rag_results", return_value=[{"question": "q1", "answer": "rag"}]),
                mock.patch.object(topk_script.compare, "_resolve_eval_mode", return_value=("local", None)),
                mock.patch.object(topk_script.compare, "evaluate_local_results", return_value=topk_script.pd.DataFrame([{"answer_relevancy": 0.9}])),
                mock.patch.object(topk_script.compare, "compute_metrics", side_effect=lambda frame, method, extra=None: {"method": method, "answer_relevancy": 0.9 if method == "RAG" else 0.6, **(extra or {})}),
                mock.patch.object(topk_script.compare, "_build_comparison_frame", return_value=topk_script.pd.DataFrame([{"method": "RAG", "answer_relevancy": 0.9}, {"method": "Direct", "answer_relevancy": 0.6}])),
                mock.patch.object(topk_script.compare, "_write_metric_plot", return_value=None),
            ):
                summary = topk_script.run_topk_from_main(
                    main_dir=main_dir,
                    data_path=data_path,
                    output_dir=output_dir,
                    candidate_top_k=128,
                    eval_mode="local",
                )

            self.assertEqual(summary["candidate_top_k"], 128)
            self.assertEqual(summary["model_count"], 1)
            copied_direct = output_dir / "models" / "deepseek-v3-2" / "round_001" / "direct_answers.jsonl"
            self.assertTrue(copied_direct.exists())
            config = json.loads((output_dir / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["candidate_top_k"], 128)
            self.assertEqual(config["top_k"], 5)


if __name__ == "__main__":
    unittest.main()
