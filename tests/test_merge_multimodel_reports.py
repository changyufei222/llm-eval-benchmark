from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


fake_pgvector_store = types.ModuleType("ragkb.storage.pgvector_store")
fake_pgvector_store.keyword_search = lambda *args, **kwargs: []
fake_pgvector_store.search = lambda *args, **kwargs: []
sys.modules.setdefault("ragkb.storage.pgvector_store", fake_pgvector_store)


SCRIPT_PATH = REPO_ROOT / "scripts" / "merge_multimodel_reports.py"
SPEC = importlib.util.spec_from_file_location("merge_multimodel_reports", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
merge_reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_reports)


class MergeMultimodelReportsTests(unittest.TestCase):
    def test_merge_reports_combines_model_dirs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            base_dir = temp / "base"
            supplement_dir = temp / "supplement"
            out_dir = temp / "merged"

            self._make_report_dir(
                base_dir,
                models=[
                    {"label": "DeepSeek-V3.2", "model": "DeepSeek-V3.2", "slug": "deepseek-v3-2"},
                    {"label": "GLM-5", "model": "GLM-5", "slug": "glm-5"},
                ],
            )
            self._make_report_dir(
                supplement_dir,
                models=[
                    {"label": "MiniMax-M2", "model": "MiniMax-M2", "slug": "minimax-m2"},
                ],
            )

            summary = merge_reports.merge_multimodel_reports(
                [base_dir, supplement_dir],
                out_dir,
            )

            self.assertEqual(summary["model_count"], 3)
            config = json.loads((out_dir / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [spec["model"] for spec in config["benchmark_models"]],
                ["DeepSeek-V3.2", "GLM-5", "MiniMax-M2"],
            )
            self.assertTrue((out_dir / "models" / "deepseek-v3-2" / "round_001" / "summary.json").exists())
            self.assertTrue((out_dir / "models" / "glm-5" / "round_001" / "summary.json").exists())
            self.assertTrue((out_dir / "models" / "minimax-m2" / "round_001" / "summary.json").exists())
            self.assertTrue((out_dir / "leaderboard.csv").exists())
            self.assertTrue((out_dir / "model_summary.csv").exists())

    def _make_report_dir(self, report_dir: Path, *, models: list[dict[str, str]]) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "models").mkdir(parents=True, exist_ok=True)
        experiment_config = {
            "data_path": "data/fbtp_eval_fixed_120.jsonl",
            "population_size": 120,
            "rounds": 1,
            "sample_size": 120,
            "with_replacement": False,
            "seed": 42,
            "benchmark_models": models,
        }
        (report_dir / "experiment_config.json").write_text(
            json.dumps(experiment_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary = {
            "data_path": "data/fbtp_eval_fixed_120.jsonl",
            "population_size": 120,
            "model_count": len(models),
            "rounds": 1,
            "sample_size": 120,
            "with_replacement": False,
            "seed": 42,
            "models": [
                {
                    "label": spec["label"],
                    "model": spec["model"],
                    "slug": spec["slug"],
                    "summary_path": str(report_dir / "models" / spec["slug"] / "summary.json"),
                }
                for spec in models
            ],
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        model_summary_lines = ["model_label,model,method,answer_relevancy_mean"]
        per_round_lines = ["model_label,model,method,round,seed,sample_size,unique_questions,answer_relevancy"]
        for spec in models:
            model_dir = report_dir / "models" / spec["slug"] / "round_001"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "summary.json").write_text(
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
            model_summary_lines.append(f"{spec['label']},{spec['model']},RAG,0.8")
            model_summary_lines.append(f"{spec['label']},{spec['model']},Direct,0.6")
            per_round_lines.append(f"{spec['label']},{spec['model']},RAG,1,42,120,120,0.8")
            per_round_lines.append(f"{spec['label']},{spec['model']},Direct,1,42,120,120,0.6")

        (report_dir / "model_summary.csv").write_text("\n".join(model_summary_lines) + "\n", encoding="utf-8")
        (report_dir / "per_round_results.csv").write_text("\n".join(per_round_lines) + "\n", encoding="utf-8")
        (report_dir / "sampling_rounds.csv").write_text("\n".join(per_round_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
