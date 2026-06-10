from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ModelSweepTests(unittest.TestCase):
    def test_slugify_model_name_normalizes_for_paths(self) -> None:
        sweep = import_module("pipelines.model_sweep")
        self.assertEqual(sweep.slugify_model_name("Qwen3.5-397B-A17B"), "qwen3-5-397b-a17b")
        self.assertEqual(sweep.slugify_model_name(" DeepSeek-V3.2 "), "deepseek-v3-2")

    def test_parse_models_text_ignores_comments_and_deduplicates(self) -> None:
        sweep = import_module("pipelines.model_sweep")
        parsed = sweep.parse_models_text(
            """
            # text-only preset
            DeepSeek-V3.2

            GLM-5
            DeepSeek-V3.2
            Kimi-K2.5
            """
        )
        self.assertEqual(parsed, ["DeepSeek-V3.2", "GLM-5", "Kimi-K2.5"])

    def test_apply_base_url_overrides_sets_openai_compatible_keys(self) -> None:
        sweep = import_module("pipelines.model_sweep")
        env = {"LLM_MODEL": "DeepSeek-V3.2"}
        sweep.apply_base_url_overrides(env, "https://api.vectorengine.cn/v1")
        self.assertEqual(env["BASE_URL"], "https://api.vectorengine.cn/v1")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://api.vectorengine.cn/v1")
        self.assertEqual(env["OPENAI_API_BASE"], "https://api.vectorengine.cn/v1")


if __name__ == "__main__":
    unittest.main()
