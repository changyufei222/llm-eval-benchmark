from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_paracloud_formal_jobs.py"
SPEC = importlib.util.spec_from_file_location("generate_paracloud_formal_jobs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_jobs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_jobs)


class GenerateParacloudFormalJobsTests(unittest.TestCase):
    def test_main_job_body_uses_controlled_8_model_list(self) -> None:
        body = generate_jobs._main_job_body(
            jobs_dir=Path("/tmp/jobs"),
            shared_scripts_dir=Path("/tmp/shared"),
            reports_root=Path("/tmp/reports"),
            time_limit="18:00:00",
        )

        self.assertIn("--benchmark-model DeepSeek-V3.2", body)
        self.assertIn("--benchmark-model GLM-5", body)
        self.assertIn("--benchmark-model Kimi-K2.5", body)
        self.assertIn("--benchmark-model DeepSeek-R1", body)
        self.assertIn("--benchmark-model Qwen3-235B-A22B-Instruct-2507", body)
        self.assertIn("--benchmark-model MiniMax-M2", body)
        self.assertIn("--benchmark-model MiniMax-M2.7", body)
        self.assertIn("--benchmark-model ERNIE-4.5-Turbo-128K", body)

    def test_sampling_job_body_keeps_anchor_4_model_stability_scope(self) -> None:
        group = {"group_name": "group_01", "round_start": 1, "round_count": 5}

        body = generate_jobs._sampling_job_body(
            jobs_dir=Path("/tmp/jobs"),
            shared_scripts_dir=Path("/tmp/shared"),
            reports_root=Path("/tmp/reports"),
            group=group,
            base_seed=42,
            time_limit="18:00:00",
        )

        self.assertIn("--benchmark-model DeepSeek-V3.2", body)
        self.assertIn("--benchmark-model GLM-5", body)
        self.assertIn("--benchmark-model Kimi-K2.5", body)
        self.assertIn("--benchmark-model DeepSeek-R1", body)
        self.assertNotIn("--benchmark-model Qwen3-235B-A22B-Instruct-2507", body)

    def test_appendix_job_body_uses_same_8_models_and_native_thinking(self) -> None:
        body = generate_jobs._appendix_job_body(
            jobs_dir=Path("/tmp/jobs"),
            shared_scripts_dir=Path("/tmp/shared"),
            reports_root=Path("/tmp/reports"),
            time_limit="18:00:00",
        )

        self.assertIn("BENCHMARK_DIRECT_THINKING_MODE=enabled", body)
        self.assertIn("RAGKB_OPENAI_THINKING_MODE=enabled", body)
        self.assertIn("--benchmark-model Qwen3-235B-A22B-Instruct-2507", body)
        self.assertIn("--benchmark-model ERNIE-4.5-Turbo-128K", body)

    def test_topk_job_body_keeps_final_top_k_fixed_and_sweeps_candidate_pool(self) -> None:
        body = generate_jobs._topk_job_body(
            jobs_dir=Path("/tmp/jobs"),
            shared_scripts_dir=Path("/tmp/shared"),
            reports_root=Path("/tmp/reports"),
            candidate_top_k=128,
            time_limit="18:00:00",
        )

        self.assertIn("--top-k 5", body)
        self.assertIn("--candidate-top-k 128", body)
        self.assertIn("--benchmark-model MiniMax-M2", body)
        self.assertIn("--benchmark-model ERNIE-4.5-Turbo-128K", body)


if __name__ == "__main__":
    unittest.main()
