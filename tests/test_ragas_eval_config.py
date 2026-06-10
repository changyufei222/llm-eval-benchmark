from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines import ragas_eval


class RagasEvalConfigTests(unittest.TestCase):
    def test_direct_metric_uses_single_generation_strictness(self) -> None:
        metrics = ragas_eval._metric_list("direct")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].name, "answer_relevancy")
        self.assertEqual(metrics[0].strictness, 1)

    def test_rag_metric_keeps_default_answer_relevancy_strictness(self) -> None:
        metrics = ragas_eval._metric_list("rag")
        answer_metric = next(metric for metric in metrics if metric.name == "answer_relevancy")
        self.assertEqual(answer_metric.strictness, 3)

    def test_metric_list_can_bind_runtime_llm_and_embeddings(self) -> None:
        llm = object()
        embeddings = object()

        metrics = ragas_eval._metric_list("rag", llm=llm, embeddings=embeddings)

        for metric in metrics:
            if hasattr(metric, "llm"):
                self.assertIs(metric.llm, llm)
            if hasattr(metric, "embeddings"):
                self.assertIs(metric.embeddings, embeddings)


if __name__ == "__main__":
    unittest.main()
