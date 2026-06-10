from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines import benchmark_protocol  # type: ignore[attr-defined]


class BenchmarkProtocolTests(unittest.TestCase):
    def test_controlled_model_sets_match_agreed_scope(self) -> None:
        self.assertEqual(
            benchmark_protocol.CONTROLLED_MAIN_MODELS,
            (
                "DeepSeek-V3.2",
                "GLM-5",
                "Kimi-K2.5",
                "DeepSeek-R1",
                "Qwen3-235B-A22B-Instruct-2507",
                "MiniMax-M2",
                "MiniMax-M2.7",
                "ERNIE-4.5-Turbo-128K",
            ),
        )
        self.assertEqual(
            benchmark_protocol.STABILITY_ANCHOR_MODELS,
            (
                "DeepSeek-V3.2",
                "GLM-5",
                "Kimi-K2.5",
                "DeepSeek-R1",
            ),
        )
        self.assertEqual(
            benchmark_protocol.APPENDIX_NATIVE_MODELS,
            benchmark_protocol.CONTROLLED_MAIN_MODELS,
        )
        self.assertEqual(
            benchmark_protocol.MISSING_MAIN_MODELS,
            (
                "Qwen3-235B-A22B-Instruct-2507",
                "MiniMax-M2",
                "MiniMax-M2.7",
                "ERNIE-4.5-Turbo-128K",
            ),
        )
        self.assertEqual(
            benchmark_protocol.MISSING_APPENDIX_MODELS,
            (
                "DeepSeek-V3.2",
                "Qwen3-235B-A22B-Instruct-2507",
                "MiniMax-M2",
                "MiniMax-M2.7",
                "ERNIE-4.5-Turbo-128K",
            ),
        )
        self.assertEqual(benchmark_protocol.TOPK_CANDIDATE_VALUES, (32, 64, 128, 256))
        self.assertEqual(benchmark_protocol.FINAL_CONTEXT_TOP_K, 5)

    def test_controlled_thinking_mode_disables_supported_main_families_only(self) -> None:
        self.assertEqual(benchmark_protocol.controlled_thinking_mode("DeepSeek-V3.2"), "disabled")
        self.assertEqual(benchmark_protocol.controlled_thinking_mode("GLM-5"), "disabled")
        self.assertEqual(benchmark_protocol.controlled_thinking_mode("Kimi-K2.5"), "disabled")
        self.assertEqual(
            benchmark_protocol.controlled_thinking_mode("Qwen3-235B-A22B-Instruct-2507"),
            "disabled",
        )
        self.assertEqual(benchmark_protocol.controlled_thinking_mode("MiniMax-M2"), "disabled")
        self.assertEqual(benchmark_protocol.controlled_thinking_mode("MiniMax-M2.7"), "disabled")
        self.assertEqual(
            benchmark_protocol.controlled_thinking_mode("ERNIE-4.5-Turbo-128K"),
            "disabled",
        )
        self.assertIsNone(benchmark_protocol.controlled_thinking_mode("DeepSeek-R1"))


if __name__ == "__main__":
    unittest.main()
