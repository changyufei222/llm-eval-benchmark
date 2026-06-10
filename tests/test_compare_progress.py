from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


fake_pgvector_store = types.ModuleType("ragkb.storage.pgvector_store")
fake_pgvector_store.keyword_search = lambda *args, **kwargs: []
fake_pgvector_store.search = lambda *args, **kwargs: []
sys.modules.setdefault("ragkb.storage.pgvector_store", fake_pgvector_store)


from pipelines import compare


def _fake_openai_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 4):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class CompareProgressTests(unittest.TestCase):
    def test_build_direct_request_kwargs_constrains_glm_direct_generation(self) -> None:
        question = "What UniProt_Entry identifier is recorded for protein PROT-00649?"

        kwargs = compare._build_direct_request_kwargs(question, model="GLM-5", request_timeout=120.0)

        self.assertEqual(kwargs["model"], "GLM-5")
        self.assertEqual(kwargs["timeout"], 120.0)
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertAlmostEqual(kwargs["temperature"], 0.01)
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertIn("Do not show reasoning", kwargs["messages"][0]["content"])
        self.assertIn("under 80 words", kwargs["messages"][0]["content"])
        self.assertEqual(kwargs["messages"][1], {"role": "user", "content": question})

    def test_build_direct_request_kwargs_disables_deepseek_v3_thinking(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is reported about the interaction between env and CD4?",
            model="DeepSeek-V3.2",
            request_timeout=120.0,
        )

        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_build_direct_request_kwargs_disables_kimi_thinking(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is reported about the interaction between env and CD4?",
            model="Kimi-K2.5",
            request_timeout=120.0,
        )

        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["max_tokens"], 128)

    def test_build_direct_request_kwargs_keeps_r1_default_thinking(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is the fixed-set composition used in benchmark phase 1?",
            model="DeepSeek-R1",
            request_timeout=120.0,
        )

        self.assertNotIn("extra_body", kwargs)

    def test_build_direct_request_kwargs_disables_qwen3_main_model_thinking(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is the fixed-set composition used in benchmark phase 1?",
            model="Qwen3-235B-A22B-Instruct-2507",
            request_timeout=120.0,
        )

        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_build_direct_request_kwargs_disables_minimax_main_model_thinking(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is the fixed-set composition used in benchmark phase 1?",
            model="MiniMax-M2",
            request_timeout=120.0,
        )

        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_build_direct_request_kwargs_omits_unsupported_baichuan_thinking_and_raises_budget(self) -> None:
        kwargs = compare._build_direct_request_kwargs(
            "What is reported about the interaction between KLK7 and CDSN?",
            model="Baichuan-M3",
            request_timeout=120.0,
        )

        self.assertNotIn("extra_body", kwargs)
        self.assertGreaterEqual(kwargs["max_tokens"], 512)

    def test_build_direct_request_kwargs_ignores_baichuan_thinking_override(self) -> None:
        with mock.patch.dict("os.environ", {"BENCHMARK_DIRECT_THINKING_MODE": "enabled"}, clear=False):
            kwargs = compare._build_direct_request_kwargs(
                "What is reported about the interaction between KLK7 and CDSN?",
                model="Baichuan-M3",
                request_timeout=120.0,
            )

        self.assertNotIn("extra_body", kwargs)
        self.assertGreaterEqual(kwargs["max_tokens"], 512)

    def test_build_rag_results_logs_row_progress(self) -> None:
        rows = [
            {
                "question": "What is reported about KLK7 and CDSN?",
                "ground_truth": "KLK7 cleaves CDSN.",
                "record_type": "ragppi",
                "tags": ["ragppi"],
            }
        ]
        settings = mock.Mock()
        contexts = [
            {
                "content": "KLK7 cleaves CDSN.",
                "score": 0.92,
                "source": "ragppi_ingest.csv",
                "metadata": {"table_name": "interaction_cards"},
            }
        ]

        with (
            mock.patch.object(compare, "retrieve", return_value=contexts),
            mock.patch.object(compare, "build_answer", return_value="KLK7 cleaves CDSN."),
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            results = compare._build_rag_results(rows, settings, top_k=5)

        self.assertEqual(len(results), 1)
        output = stdout.getvalue()
        self.assertIn("benchmark_rag_row_start row='1/1'", output)
        self.assertIn("benchmark_rag_retrieve_done row='1/1'", output)
        self.assertIn("benchmark_rag_answer_done row='1/1'", output)

    def test_build_rag_results_retries_answer_and_persists_final_row(self) -> None:
        rows = [
            {
                "question": "What affinity is reported for INT-00001?",
                "ground_truth": "6.2 nM",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            }
        ]
        settings = mock.Mock()
        contexts = [
            {
                "content": "Affinity is 6.2 nM.",
                "score": 0.95,
                "source": "interaction_cards.jsonl",
                "metadata": {"table_name": "affinity_data"},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "rag_answers.jsonl"
            with (
                mock.patch.dict("os.environ", {"BENCHMARK_ROW_MAX_RETRIES": "1"}, clear=False),
                mock.patch.object(compare, "retrieve", return_value=contexts),
                mock.patch.object(compare, "build_answer", side_effect=[TimeoutError("first timeout"), "Affinity is 6.2 nM."]),
                mock.patch("time.sleep", return_value=None),
            ):
                results = compare._build_rag_results(rows, settings, top_k=5, persist_path=persist_path)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "ok")
            self.assertEqual(results[0]["attempts"], 2)
            persisted = [json.loads(line) for line in persist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0]["attempts"], 2)

    def test_build_rag_results_records_failure_and_continues(self) -> None:
        rows = [
            {
                "question": "Question one",
                "ground_truth": "Answer one",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            },
            {
                "question": "Question two",
                "ground_truth": "Answer two",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            },
        ]
        settings = mock.Mock()
        contexts = [
            {
                "content": "Context",
                "score": 0.9,
                "source": "cards.jsonl",
                "metadata": {"table_name": "cards"},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "rag_answers.jsonl"
            with (
                mock.patch.dict("os.environ", {"BENCHMARK_ROW_MAX_RETRIES": "0"}, clear=False),
                mock.patch.object(compare, "retrieve", return_value=contexts),
                mock.patch.object(compare, "build_answer", side_effect=[TimeoutError("row one timeout"), "Answer two"]),
                mock.patch("time.sleep", return_value=None),
            ):
                results = compare._build_rag_results(rows, settings, top_k=5, persist_path=persist_path)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["status"], "answer_error")
            self.assertEqual(results[1]["status"], "ok")
            persisted = [json.loads(line) for line in persist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["status"] for row in persisted], ["answer_error", "ok"])

    def test_build_direct_results_retries_answer_and_persists_final_row(self) -> None:
        rows = [
            {
                "question": "What is reported for INT-00001?",
                "ground_truth": "6.2 nM",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            }
        ]
        client = mock.Mock()
        client.chat.completions.create = mock.Mock(
            side_effect=[TimeoutError("first timeout"), _fake_openai_response("Affinity is 6.2 nM.", 12, 5)]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "direct_answers.jsonl"
            with (
                mock.patch.dict("os.environ", {"BENCHMARK_ROW_MAX_RETRIES": "1"}, clear=False),
                mock.patch.object(compare, "OpenAI", return_value=client),
                mock.patch("time.sleep", return_value=None),
            ):
                results = compare._build_direct_results_openai(rows, model="test-model", persist_path=persist_path)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "ok")
            self.assertEqual(results[0]["attempts"], 2)
            persisted = [json.loads(line) for line in persist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0]["attempts"], 2)

    def test_build_direct_results_records_failure_and_continues(self) -> None:
        rows = [
            {
                "question": "Question one",
                "ground_truth": "Answer one",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            },
            {
                "question": "Question two",
                "ground_truth": "Answer two",
                "record_type": "jsonl",
                "tags": ["ragppi"],
            },
        ]
        client = mock.Mock()
        client.chat.completions.create = mock.Mock(
            side_effect=[TimeoutError("row one timeout"), _fake_openai_response("Answer two", 11, 3)]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "direct_answers.jsonl"
            with (
                mock.patch.dict("os.environ", {"BENCHMARK_ROW_MAX_RETRIES": "0"}, clear=False),
                mock.patch.object(compare, "OpenAI", return_value=client),
                mock.patch("time.sleep", return_value=None),
            ):
                results = compare._build_direct_results_openai(rows, model="test-model", persist_path=persist_path)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["status"], "answer_error")
            self.assertEqual(results[1]["status"], "ok")
            persisted = [json.loads(line) for line in persist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["status"] for row in persisted], ["answer_error", "ok"])


if __name__ == "__main__":
    unittest.main()
