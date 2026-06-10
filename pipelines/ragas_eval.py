from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from pipelines._bootstrap_ragkb import ensure_ragkb_on_path

ensure_ragkb_on_path()

from metrics.metrics import compute_metrics, frame_to_markdown, metrics_to_markdown
from metrics.plotting import write_metric_plot
from pipelines.ragas_json_compat import install_ragas_fenced_json_compat
from ragkb.config import Settings


class LocalHashEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed_query(self, text: str) -> List[float]:
        from ragkb.embeddings import embed_texts

        return embed_texts([text], self.settings)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from ragkb.embeddings import embed_texts

        return embed_texts(texts, self.settings)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_base_url() -> str | None:
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "BASE_URL"):
        value = os.getenv(key)
        if value:
            return value.rstrip("/")
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _build_dataset(rows: List[Dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "contexts": row.get("contexts", []),
                "ground_truth": row.get("ground_truth", ""),
            }
            for row in rows
        ]
    )


def _answer_relevancy_metric(strictness: int = 3) -> object:
    metric = copy.deepcopy(answer_relevancy)
    metric.strictness = strictness
    return metric


def _bind_metric_runtime(metric: object, llm: object | None = None, embeddings: object | None = None) -> object:
    metric = copy.deepcopy(metric)
    if hasattr(metric, "llm"):
        metric.llm = llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = embeddings
    return metric


def _metric_list(metric_set: str, llm: object | None = None, embeddings: object | None = None) -> List[object]:
    if metric_set == "direct":
        return [_bind_metric_runtime(_answer_relevancy_metric(strictness=1), llm=llm, embeddings=embeddings)]
    return [
        _bind_metric_runtime(faithfulness, llm=llm, embeddings=embeddings),
        _bind_metric_runtime(_answer_relevancy_metric(strictness=3), llm=llm, embeddings=embeddings),
        _bind_metric_runtime(context_precision, llm=llm, embeddings=embeddings),
    ]


def _has_signal(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return False
    return not numeric.isna().all().all()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.ragas_eval")
    parser.add_argument("--answers-path", required=True)
    parser.add_argument("--metric-set", choices=["rag", "direct"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scores-name", required=True)
    parser.add_argument("--summary-name", required=True)
    args = parser.parse_args()

    load_dotenv()
    install_ragas_fenced_json_compat()
    settings = Settings()

    answers_path = Path(args.answers_path)
    if not answers_path.exists():
        raise FileNotFoundError(str(answers_path))

    rows = _load_jsonl(answers_path)
    dataset = _build_dataset(rows)

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
    metrics = _metric_list(args.metric_set, llm=llm, embeddings=embeddings)
    run_config = RunConfig(
        timeout=_env_int("RAGAS_TIMEOUT", 180),
        max_retries=_env_int("RAGAS_MAX_RETRIES", 10),
        max_wait=_env_int("RAGAS_MAX_WAIT", 60),
        max_workers=_env_int("RAGAS_MAX_WORKERS", 16),
    )
    batch_size = _env_int("RAGAS_BATCH_SIZE", 15)
    df = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=batch_size,
    ).to_pandas()

    if not _has_signal(df):
        raise RuntimeError("ragas_empty_results")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_base = Path(args.scores_name)
    summary_base = Path(args.summary_name)
    df.to_csv(out_dir / f"{scores_base}.csv", index=False)
    (out_dir / f"{scores_base}.md").write_text(frame_to_markdown(df), encoding="utf-8")

    extra = {
        "metric_set": args.metric_set,
        "model": settings.llm_model,
        "answers_path": str(answers_path),
    }
    summary = compute_metrics(df, method=args.metric_set.upper(), extra=extra)
    (out_dir / f"{summary_base}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{summary_base}.md").write_text(
        metrics_to_markdown(f"# {args.metric_set.upper()} RAGAS Summary", summary),
        encoding="utf-8",
    )
    write_metric_plot(pd.DataFrame([summary]), out_dir / f"{summary_base}.png", title=f"{args.metric_set.upper()} RAGAS Summary")
    print(f"Saved ragas {args.metric_set} outputs to {out_dir}")


if __name__ == "__main__":
    main()
