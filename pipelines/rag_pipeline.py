from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pipelines._bootstrap_ragkb import ensure_ragkb_on_path

ensure_ragkb_on_path()

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install ragas and datasets to run this pipeline") from exc

try:
    from ragkb.answer.generator import build_answer
    from ragkb.config import Settings
    from ragkb.retrieval.retriever import retrieve
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ragkb package not found. Install with: pip install -e ../llm-rag-knowledge-base"
    ) from exc

from metrics.metrics import compute_metrics, metrics_to_markdown
from pipelines.ragas_eval import LocalHashEmbeddings, _metric_list, _resolve_base_url
from pipelines.ragas_json_compat import install_ragas_fenced_json_compat


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.rag_pipeline")
    parser.add_argument("--data-path", default="data/fbtp_eval.jsonl")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    install_ragas_fenced_json_compat()
    settings = Settings()
    settings.evidence_mode = "none"

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    rows = _load_jsonl(data_path)
    results = []

    for row in rows:
        question = row.get("question", "")
        ground_truth = row.get("ground_truth", "")
        contexts = retrieve(question, top_k=args.top_k, settings=settings)
        answer = build_answer(question, contexts, settings)
        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [context.get("content", "") for context in contexts],
                "ground_truth": ground_truth,
                "context_count": len(contexts),
            }
        )

    dataset = Dataset.from_list(results)
    llm_kwargs = {
        "model": settings.llm_model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "request_timeout": 120.0,
        "max_retries": 1,
        "temperature": 0.01,
    }
    base_url = _resolve_base_url()
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)
    embeddings = LocalHashEmbeddings(settings)
    metrics = _metric_list("rag", llm=llm, embeddings=embeddings)

    report = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = report.to_pandas()
    df.to_csv(out_dir / "ragas_scores.csv", index=False)
    df.to_markdown(out_dir / "ragas_scores.md", index=False)
    _write_jsonl(out_dir / "rag_answers.jsonl", results)

    summary = compute_metrics(
        df,
        method="RAG",
        extra={"samples": len(df), "data_path": str(data_path)},
    )
    (out_dir / "ragas_summary.md").write_text(
        metrics_to_markdown("# RAGAS Summary", summary),
        encoding="utf-8",
    )
    (out_dir / "ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    numeric = df.select_dtypes(include="number")
    mean_scores = numeric.mean().sort_values(ascending=False) if not numeric.empty else pd.Series(dtype=float)
    if not mean_scores.empty:
        plt.figure(figsize=(8, 4))
        mean_scores.plot(kind="bar")
        plt.title("RAGAS Mean Metrics")
        plt.ylabel("score")
        plt.tight_layout()
        plt.savefig(out_dir / "ragas_summary.png", dpi=150)
        plt.close()

    print("Saved RAGAS report to", out_dir)


if __name__ == "__main__":
    main()
