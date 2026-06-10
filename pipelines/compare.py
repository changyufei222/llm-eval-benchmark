from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
from typing import Dict, List, Sequence
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pipelines._bootstrap_ragkb import ensure_ragkb_on_path

ensure_ragkb_on_path()

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    ChatOpenAI = None
    LANGCHAIN_OPENAI_IMPORT_ERROR = exc

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    RAGAS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    Dataset = None
    evaluate = None
    answer_relevancy = None
    context_precision = None
    faithfulness = None
    RAGAS_IMPORT_ERROR = exc

try:
    from ragkb.answer.generator import build_answer
    from ragkb.config import Settings
    from ragkb.openai_compat import adjusted_completion_max_tokens, openai_thinking_mode
    from ragkb.retrieval.retriever import retrieve
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ragkb package not found. Install with: pip install -e ../llm-rag-knowledge-base"
    ) from exc

from openai import OpenAI

from metrics.metrics import (
    build_category_breakdown,
    compute_metrics,
    estimate_row_cost,
    evaluate_local_results,
    frame_to_markdown,
    list_hit_rate,
    metrics_to_markdown,
    summarize_grouped_distribution,
)
from metrics.fallback import build_fallback_notice
from metrics.plotting import write_metric_plot
def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _LocalHashEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed_query(self, text: str) -> List[float]:
        from ragkb.embeddings import embed_texts

        return embed_texts([text], self.settings)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from ragkb.embeddings import embed_texts

        return embed_texts(texts, self.settings)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _resolve_base_url() -> str | None:
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "BASE_URL"):
        value = os.getenv(key)
        if value:
            return value.rstrip("/")
    return None


def _resolve_endpoint() -> tuple[str, int]:
    base_url = _resolve_base_url()
    if not base_url:
        return "api.openai.com", 443

    parsed = urlparse(base_url)
    host = parsed.hostname or "api.openai.com"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _openai_reachable(timeout: float = 5.0) -> bool:
    host, port = _resolve_endpoint()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_eval_mode(requested: str) -> tuple[str, str | None]:
    if requested == "local":
        return "local", "forced_local"

    if RAGAS_IMPORT_ERROR is not None:
        if requested == "ragas":
            raise RuntimeError("ragas is unavailable in the current environment") from RAGAS_IMPORT_ERROR
        return "local", f"ragas_unavailable:{RAGAS_IMPORT_ERROR.__class__.__name__}"

    if not _openai_reachable():
        if requested == "ragas":
            raise RuntimeError("OpenAI API is unreachable from this machine")
        return "local", "openai_unreachable"

    return "ragas", None


def _context_source_markers(contexts: Sequence[Dict]) -> List[str]:
    markers: List[str] = []
    keys = [
        "Sources_identifier",
        "Sources_title",
        "File_Name",
        "\ufeffFile_Name",
        "source_path",
    ]
    for context in contexts:
        markers.append(str(context.get("source", "")))
        meta = context.get("metadata", {}) or {}
        for key in keys:
            value = meta.get(key)
            if value:
                markers.append(str(value))
    return [marker for marker in markers if marker]


def _coerce_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _context_table_markers(contexts: Sequence[Dict]) -> List[str]:
    markers: List[str] = []
    for context in contexts:
        meta = context.get("metadata", {}) or {}
        table_name = meta.get("table_name")
        if table_name:
            markers.append(str(table_name))
        markers.extend(_coerce_list(meta.get("table_sources")))
    return [marker for marker in markers if marker]


def _context_id_markers(contexts: Sequence[Dict]) -> List[str]:
    markers: List[str] = []
    scalar_keys = [
        "primary_key_value",
        "protein_id",
        "interaction_id",
        "domain_id",
        "target_variant_id",
        "source_id",
    ]
    list_keys = ["domain_ids", "interaction_ids"]
    for context in contexts:
        meta = context.get("metadata", {}) or {}
        for key in scalar_keys:
            value = meta.get(key)
            if value not in (None, ""):
                markers.append(str(value))
        for key in list_keys:
            markers.extend(_coerce_list(meta.get(key)))
        foreign_keys = meta.get("foreign_keys") or {}
        if isinstance(foreign_keys, dict):
            markers.extend(_coerce_list(list(foreign_keys.values())))
    return [marker for marker in markers if marker]


def _primary_category(row: Dict) -> str:
    tags = row.get("tags") or []
    if isinstance(tags, list) and tags:
        return str(tags[0])
    if isinstance(tags, str) and tags.strip():
        return tags.split(",")[0].strip()
    return str(row.get("record_type") or "unknown")


def _rough_token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text or "")))


def _pricing() -> tuple[float, float]:
    return (
        float(os.getenv("BENCHMARK_INPUT_COST_PER_MILLION", "0")),
        float(os.getenv("BENCHMARK_OUTPUT_COST_PER_MILLION", "0")),
    )


def _progress_print(event: str, **fields: object) -> None:
    rendered = " ".join(f"{key}={str(value)!r}" for key, value in fields.items())
    print(f"{time.strftime('%F %T')} {event} {rendered}".rstrip(), flush=True)


def _question_preview(question: str, limit: int = 80) -> str:
    normalized = " ".join(str(question).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _row_retry_count() -> int:
    return max(0, int(os.getenv("BENCHMARK_ROW_MAX_RETRIES", "1")))


def _row_retry_backoff_seconds() -> float:
    return max(0.0, float(os.getenv("BENCHMARK_ROW_RETRY_BACKOFF_SECONDS", "2.0")))


def _direct_system_prompt() -> str:
    return (
        "You are answering benchmark questions without retrieval context. "
        "Return only the shortest factual answer to the user's question. "
        "Do not show reasoning, analysis, or chain-of-thought. "
        "Do not think step-by-step. "
        "Keep the answer under 80 words and at most 2 short sentences. "
        "For identifier or field lookup questions, return only the requested field or value. "
        "If you are unsure, say \"I don't know.\""
    )


def _direct_max_tokens() -> int:
    return max(1, int(os.getenv("BENCHMARK_DIRECT_MAX_TOKENS", "128")))


def _direct_temperature() -> float:
    return max(0.0, float(os.getenv("BENCHMARK_DIRECT_TEMPERATURE", "0.01")))


def _normalized_model_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(model).lower())


def _direct_thinking_mode(model: str) -> str | None:
    override = os.getenv("BENCHMARK_DIRECT_THINKING_MODE", "").strip().lower()
    if override not in {"enabled", "disabled"}:
        override = None
    return openai_thinking_mode(model, override=override)


def _build_direct_request_kwargs(question: str, *, model: str, request_timeout: float) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _direct_system_prompt()},
            {"role": "user", "content": question},
        ],
        "timeout": request_timeout,
        "temperature": _direct_temperature(),
        "max_tokens": adjusted_completion_max_tokens(model, _direct_max_tokens()),
    }
    thinking_mode = _direct_thinking_mode(model)
    if thinking_mode:
        kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
    return kwargs


def _append_jsonl_row(path: Path | None, row: Dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _error_excerpt(exc: Exception, limit: int = 240) -> str:
    message = f"{exc.__class__.__name__}: {exc}".strip()
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _build_rag_row_result(
    row: Dict,
    *,
    record_type: str | None,
    answer: str,
    contexts: Sequence[Dict],
    latency_ms: float,
    attempts: int,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> Dict:
    question = row.get("question", "")
    ground_truth = row.get("ground_truth", "")
    prompt_tokens = _rough_token_count(question) + sum(_rough_token_count(context.get("content", "")) for context in contexts)
    completion_tokens = _rough_token_count(answer)
    source_markers = _context_source_markers(contexts)
    table_markers = _context_table_markers(contexts)
    id_markers = _context_id_markers(contexts)
    input_cost, output_cost = _pricing()
    return {
        "question": question,
        "category": _primary_category(row),
        "record_type": record_type,
        "tags": row.get("tags", []),
        "answer": answer,
        "contexts": [context.get("content", "") for context in contexts],
        "ground_truth": ground_truth,
        "context_count": len(contexts),
        "context_scores": [context.get("score", 0.0) for context in contexts],
        "source_markers": source_markers,
        "table_markers": table_markers,
        "id_markers": id_markers,
        "expected_answer_contains": row.get("expected_answer_contains", []),
        "expected_source_contains": row.get("expected_source_contains", []),
        "expected_table_contains": row.get("expected_table_contains", []),
        "expected_primary_ids": row.get("expected_primary_ids", []),
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimate_row_cost(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            input_cost,
            output_cost,
        ),
        "source_hit_rate": list_hit_rate(source_markers, row.get("expected_source_contains", [])),
        "table_hit_rate": list_hit_rate(table_markers, row.get("expected_table_contains", [])),
        "id_hit_rate": list_hit_rate(id_markers, row.get("expected_primary_ids", [])),
        "status": status,
        "attempts": attempts,
        "error_type": error_type or "",
        "error_message": error_message or "",
    }


def _build_direct_row_result(
    row: Dict,
    *,
    answer: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
    attempts: int,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> Dict:
    return {
        "question": row.get("question", ""),
        "category": _primary_category(row),
        "record_type": row.get("record_type"),
        "tags": row.get("tags", []),
        "answer": answer,
        "contexts": [],
        "ground_truth": row.get("ground_truth", ""),
        "context_count": 0,
        "context_scores": [],
        "source_markers": [],
        "table_markers": [],
        "id_markers": [],
        "expected_answer_contains": row.get("expected_answer_contains", []),
        "expected_source_contains": row.get("expected_source_contains", []),
        "expected_table_contains": row.get("expected_table_contains", []),
        "expected_primary_ids": row.get("expected_primary_ids", []),
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "source_hit_rate": 0.0,
        "table_hit_rate": 0.0,
        "id_hit_rate": 0.0,
        "status": status,
        "attempts": attempts,
        "error_type": error_type or "",
        "error_message": error_message or "",
    }


def _build_rag_results(
    rows: Sequence[Dict],
    settings: Settings,
    top_k: int,
    persist_path: Path | None = None,
) -> List[Dict]:
    results = []
    settings.evidence_mode = "none"
    os.environ.setdefault("RAGKB_OPENAI_PROGRESS_LOG", "1")
    total_rows = len(rows)
    max_retries = _row_retry_count()
    retry_backoff_seconds = _row_retry_backoff_seconds()
    for row_index, row in enumerate(rows, start=1):
        question = row.get("question", "")
        record_type = row.get("record_type")
        row_started = time.perf_counter()
        _progress_print(
            "benchmark_rag_row_start",
            row=f"{row_index}/{total_rows}",
            record_type=record_type,
            question=_question_preview(question),
        )
        contexts: list[Dict] = []
        try:
            contexts = retrieve(question, top_k=top_k, settings=settings, record_type=record_type)
            retrieve_ms = round((time.perf_counter() - row_started) * 1000, 2)
            _progress_print(
                "benchmark_rag_retrieve_done",
                row=f"{row_index}/{total_rows}",
                contexts=len(contexts),
                retrieve_ms=retrieve_ms,
                question=_question_preview(question),
            )
        except Exception as exc:
            result = _build_rag_row_result(
                row,
                record_type=record_type,
                answer="",
                contexts=[],
                latency_ms=round((time.perf_counter() - row_started) * 1000, 2),
                attempts=1,
                status="retrieve_error",
                error_type=exc.__class__.__name__,
                error_message=_error_excerpt(exc),
            )
            _progress_print(
                "benchmark_rag_row_failed",
                row=f"{row_index}/{total_rows}",
                stage="retrieve",
                error=_error_excerpt(exc),
                question=_question_preview(question),
            )
            results.append(result)
            _append_jsonl_row(persist_path, result)
            continue

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                answer = build_answer(question, contexts, settings)
                latency_ms = round((time.perf_counter() - row_started) * 1000, 2)
                _progress_print(
                    "benchmark_rag_answer_done",
                    row=f"{row_index}/{total_rows}",
                    total_ms=latency_ms,
                    answer_chars=len(answer),
                    attempts=attempt,
                    question=_question_preview(question),
                )
                result = _build_rag_row_result(
                    row,
                    record_type=record_type,
                    answer=answer,
                    contexts=contexts,
                    latency_ms=latency_ms,
                    attempts=attempt,
                    status="ok",
                )
                results.append(result)
                _append_jsonl_row(persist_path, result)
                break
            except Exception as exc:
                last_exc = exc
                if attempt <= max_retries:
                    _progress_print(
                        "benchmark_rag_row_retry",
                        row=f"{row_index}/{total_rows}",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        stage="answer",
                        error=_error_excerpt(exc),
                        question=_question_preview(question),
                    )
                    if retry_backoff_seconds > 0:
                        time.sleep(retry_backoff_seconds)
                    continue

                result = _build_rag_row_result(
                    row,
                    record_type=record_type,
                    answer="",
                    contexts=contexts,
                    latency_ms=round((time.perf_counter() - row_started) * 1000, 2),
                    attempts=attempt,
                    status="answer_error",
                    error_type=exc.__class__.__name__,
                    error_message=_error_excerpt(exc),
                )
                _progress_print(
                    "benchmark_rag_row_failed",
                    row=f"{row_index}/{total_rows}",
                    attempts=attempt,
                    stage="answer",
                    error=_error_excerpt(exc),
                    question=_question_preview(question),
                )
                results.append(result)
                _append_jsonl_row(persist_path, result)
                break
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected retry loop exit for question: {question}") from last_exc
    return results


def _build_local_direct_answer(question: str) -> str:
    match = re.search(r"interaction between (.+?) and (.+?)\?", question, flags=re.IGNORECASE)
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip()
        return (
            "Local direct baseline without retrieval. "
            f"The question asks about the interaction between {left} and {right}, "
            "but no supporting context was available in this mode."
        )
    return "Local direct baseline without retrieval. No supporting context was available in this mode."


def _build_direct_results_local(rows: Sequence[Dict]) -> List[Dict]:
    results = []
    for row in rows:
        question = row.get("question", "")
        answer = _build_local_direct_answer(question)
        prompt_tokens = _rough_token_count(question)
        completion_tokens = _rough_token_count(answer)
        results.append(
            {
                "question": question,
                "category": _primary_category(row),
                "record_type": row.get("record_type"),
                "tags": row.get("tags", []),
                "answer": answer,
                "contexts": [],
                "ground_truth": row.get("ground_truth", ""),
                "context_count": 0,
                "context_scores": [],
                "source_markers": [],
                "table_markers": [],
                "id_markers": [],
                "expected_answer_contains": row.get("expected_answer_contains", []),
                "expected_source_contains": row.get("expected_source_contains", []),
                "expected_table_contains": row.get("expected_table_contains", []),
                "expected_primary_ids": row.get("expected_primary_ids", []),
                "latency_ms": 0.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": 0.0,
                "source_hit_rate": 0.0,
                "table_hit_rate": 0.0,
                "id_hit_rate": 0.0,
            }
        )
    return results


def _build_direct_results_openai(
    rows: Sequence[Dict],
    model: str,
    persist_path: Path | None = None,
) -> List[Dict]:
    kwargs = {"timeout": 120.0, "max_retries": 1}
    base_url = _resolve_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    results = []
    input_cost, output_cost = _pricing()
    total_rows = len(rows)
    request_timeout = float(os.getenv("BENCHMARK_OPENAI_REQUEST_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "120")))
    max_retries = _row_retry_count()
    retry_backoff_seconds = _row_retry_backoff_seconds()
    for row_index, row in enumerate(rows, start=1):
        question = row.get("question", "")
        row_started = time.perf_counter()
        _progress_print(
            "benchmark_direct_row_start",
            row=f"{row_index}/{total_rows}",
            model=model,
            question=_question_preview(question),
        )
        for attempt in range(1, max_retries + 2):
            try:
                response = client.chat.completions.create(
                    **_build_direct_request_kwargs(question, model=model, request_timeout=request_timeout)
                )
                latency_ms = round((time.perf_counter() - row_started) * 1000, 2)
                _progress_print(
                    "benchmark_direct_row_done",
                    row=f"{row_index}/{total_rows}",
                    model=model,
                    latency_ms=latency_ms,
                    attempts=attempt,
                    question=_question_preview(question),
                )
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None) or _rough_token_count(question)
                completion_tokens = getattr(usage, "completion_tokens", None) or _rough_token_count(response.choices[0].message.content or "")
                result = _build_direct_row_result(
                    row,
                    answer=(response.choices[0].message.content or "").strip(),
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=estimate_row_cost(
                        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                        input_cost,
                        output_cost,
                    ),
                    attempts=attempt,
                    status="ok",
                )
                results.append(result)
                _append_jsonl_row(persist_path, result)
                break
            except Exception as exc:
                if attempt <= max_retries:
                    _progress_print(
                        "benchmark_direct_row_retry",
                        row=f"{row_index}/{total_rows}",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        model=model,
                        error=_error_excerpt(exc),
                        question=_question_preview(question),
                    )
                    if retry_backoff_seconds > 0:
                        time.sleep(retry_backoff_seconds)
                    continue

                latency_ms = round((time.perf_counter() - row_started) * 1000, 2)
                result = _build_direct_row_result(
                    row,
                    answer="",
                    latency_ms=latency_ms,
                    prompt_tokens=_rough_token_count(question),
                    completion_tokens=0,
                    estimated_cost_usd=0.0,
                    attempts=attempt,
                    status="answer_error",
                    error_type=exc.__class__.__name__,
                    error_message=_error_excerpt(exc),
                )
                _progress_print(
                    "benchmark_direct_row_failed",
                    row=f"{row_index}/{total_rows}",
                    attempts=attempt,
                    model=model,
                    error=_error_excerpt(exc),
                    question=_question_preview(question),
                )
                results.append(result)
                _append_jsonl_row(persist_path, result)
                break
    return results


def _evaluate_with_ragas(results: Sequence[Dict], metrics: Sequence[object]) -> pd.DataFrame:
    if Dataset is None or evaluate is None:
        raise RuntimeError("ragas is not available")
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai is not available") from LANGCHAIN_OPENAI_IMPORT_ERROR

    base_url = _resolve_base_url()
    settings = Settings()
    llm_kwargs = {
        "model": settings.llm_model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "max_retries": 1,
        "request_timeout": 120.0,
        "temperature": 0.01,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    ragas_llm = ChatOpenAI(**llm_kwargs)
    ragas_embeddings = _LocalHashEmbeddings(settings)

    dataset = Dataset.from_list(
        [
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "contexts": row.get("contexts", []),
                "ground_truth": row.get("ground_truth", ""),
            }
            for row in results
        ]
    )
    report = evaluate(
        dataset,
        metrics=list(metrics),
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return report.to_pandas()


def _ragas_has_signal(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return False
    return not numeric.isna().all().all()


def _run_ragas_subprocess(
    answers_path: Path,
    out_dir: Path,
    metric_set: str,
    model: str,
) -> tuple[pd.DataFrame, Dict]:
    if metric_set == "rag":
        scores_name = "ragas_scores"
        summary_name = "ragas_summary"
    else:
        scores_name = "direct_ragas_scores"
        summary_name = "direct_ragas_summary"

    cmd = [
        sys.executable,
        "-m",
        "pipelines.ragas_eval",
        "--answers-path",
        str(answers_path),
        "--metric-set",
        metric_set,
        "--output-dir",
        str(out_dir),
        "--scores-name",
        scores_name,
        "--summary-name",
        summary_name,
    ]
    env = os.environ.copy()
    env["LLM_MODEL"] = model
    completed = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())

    df = pd.read_csv(out_dir / f"{scores_name}.csv")
    summary = _load_json(out_dir / f"{summary_name}.json")
    return df, summary


def _build_comparison_frame(eval_mode: str, rag_summary: Dict[str, float], direct_summary: Dict[str, float]) -> pd.DataFrame:
    if eval_mode == "local":
        rows = [
            {
                "method": "RAG",
                "answer_token_f1": rag_summary.get("answer_token_f1"),
                "keyword_hit_rate": rag_summary.get("keyword_hit_rate"),
                "source_hit_rate": rag_summary.get("source_hit_rate"),
                "table_hit_rate": rag_summary.get("table_hit_rate"),
                "id_hit_rate": rag_summary.get("id_hit_rate"),
                "context_count": rag_summary.get("context_count"),
                "latency_ms": rag_summary.get("latency_ms"),
                "estimated_cost_usd": rag_summary.get("estimated_cost_usd"),
            },
            {
                "method": "Direct",
                "answer_token_f1": direct_summary.get("answer_token_f1"),
                "keyword_hit_rate": direct_summary.get("keyword_hit_rate"),
                "source_hit_rate": direct_summary.get("source_hit_rate"),
                "table_hit_rate": direct_summary.get("table_hit_rate"),
                "id_hit_rate": direct_summary.get("id_hit_rate"),
                "context_count": direct_summary.get("context_count"),
                "latency_ms": direct_summary.get("latency_ms"),
                "estimated_cost_usd": direct_summary.get("estimated_cost_usd"),
            },
        ]
        return pd.DataFrame(rows)

    rows = [
        {
            "method": "RAG",
            "faithfulness": rag_summary.get("faithfulness"),
            "answer_relevancy": rag_summary.get("answer_relevancy"),
            "context_precision": rag_summary.get("context_precision"),
            "source_hit_rate": rag_summary.get("source_hit_rate"),
            "table_hit_rate": rag_summary.get("table_hit_rate"),
            "id_hit_rate": rag_summary.get("id_hit_rate"),
            "latency_ms": rag_summary.get("latency_ms"),
            "estimated_cost_usd": rag_summary.get("estimated_cost_usd"),
        },
        {
            "method": "Direct",
            "faithfulness": None,
            "answer_relevancy": direct_summary.get("answer_relevancy"),
            "context_precision": None,
            "source_hit_rate": direct_summary.get("source_hit_rate"),
            "table_hit_rate": direct_summary.get("table_hit_rate"),
            "id_hit_rate": direct_summary.get("id_hit_rate"),
            "latency_ms": direct_summary.get("latency_ms"),
            "estimated_cost_usd": direct_summary.get("estimated_cost_usd"),
        },
    ]
    return pd.DataFrame(rows)


def _operational_summary(results: Sequence[Dict], method: str) -> Dict[str, object]:
    rows = []
    failed_rows = 0
    for row in results:
        if str(row.get("status", "ok")) != "ok":
            failed_rows += 1
        rows.append(
            {
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
                "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
                "source_hit_rate": float(row.get("source_hit_rate", 0.0) or 0.0),
                "table_hit_rate": float(row.get("table_hit_rate", 0.0) or 0.0),
                "id_hit_rate": float(row.get("id_hit_rate", 0.0) or 0.0),
            }
        )
    summary = compute_metrics(rows, method=method)
    summary["failed_rows"] = failed_rows
    summary["success_rows"] = int(len(results) - failed_rows)
    return summary


def _write_metric_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    write_metric_plot(frame, output_path, title)


def _attach_metadata(frame: pd.DataFrame, results: Sequence[Dict]) -> pd.DataFrame:
    if frame.empty:
        return frame
    metadata_rows = []
    for row in results:
        metadata_rows.append(
            {
                "question": row.get("question", ""),
                "category": row.get("category", row.get("record_type", "unknown")),
                "latency_ms": row.get("latency_ms", 0.0),
                "prompt_tokens": row.get("prompt_tokens", 0),
                "completion_tokens": row.get("completion_tokens", 0),
                "estimated_cost_usd": row.get("estimated_cost_usd", 0.0),
                "source_hit_rate": row.get("source_hit_rate", 0.0),
                "table_hit_rate": row.get("table_hit_rate", 0.0),
                "id_hit_rate": row.get("id_hit_rate", 0.0),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    if "question" in frame.columns:
        merged = frame.merge(metadata, on="question", how="left")
        return merged
    enriched = frame.copy()
    for column in metadata.columns:
        if column == "question":
            continue
        enriched[column] = metadata[column]
    return enriched


def _sample_rows(rows: Sequence[Dict], sample_size: int, with_replacement: bool, seed: int) -> List[Dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not rows:
        raise ValueError("rows must not be empty")
    if not with_replacement and sample_size > len(rows):
        raise ValueError("sample_size cannot exceed population size when sampling without replacement")

    rng = random.Random(seed)
    if with_replacement:
        return [dict(rows[rng.randrange(len(rows))]) for _ in range(sample_size)]

    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [dict(rows[index]) for index in indices[:sample_size]]


def _slugify_label(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "model"


def _parse_benchmark_model_specs(raw_specs: Sequence[str], default_model: str) -> List[Dict[str, str]]:
    specs: list[Dict[str, str]] = []
    seen_slugs: set[str] = set()
    for raw_spec in raw_specs:
        text = (raw_spec or "").strip()
        if not text:
            continue
        if "=" in text:
            label, model_name = text.split("=", 1)
            label = label.strip()
            model_name = model_name.strip()
        else:
            model_name = text
            label = text
        if not label:
            label = model_name or default_model
        if not model_name:
            model_name = default_model
        slug = _slugify_label(label)
        suffix = 2
        while slug in seen_slugs:
            slug = f"{_slugify_label(label)}-{suffix}"
            suffix += 1
        seen_slugs.add(slug)
        specs.append({"label": label, "model": model_name, "slug": slug})
    return specs


def _settings_with_model(settings: Settings, model_name: str) -> Settings:
    try:
        return replace(settings, llm_model=model_name)
    except TypeError:
        setattr(settings, "llm_model", model_name)
        return settings


def _settings_with_candidate_top_k(settings: Settings, candidate_top_k: int | None) -> Settings:
    if candidate_top_k is None:
        return settings
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive")
    try:
        return replace(settings, reranker_top_n=candidate_top_k)
    except TypeError:
        setattr(settings, "reranker_top_n", candidate_top_k)
        return settings


def _candidate_top_k_value(settings: Settings) -> int | None:
    value = getattr(settings, "reranker_top_n", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_compare_once(
    rows: Sequence[Dict],
    out_dir: Path,
    settings: Settings,
    top_k: int,
    eval_mode_requested: str,
    direct_model: str,
    fail_on_fallback: bool,
    data_path_label: str,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rag_answers_path = out_dir / "rag_answers.jsonl"
    direct_answers_path = out_dir / "direct_answers.jsonl"
    for path in (rag_answers_path, direct_answers_path):
        if path.exists():
            path.unlink()

    eval_mode, mode_reason = _resolve_eval_mode(eval_mode_requested)
    rag_results = _build_rag_results(rows, settings, top_k, persist_path=rag_answers_path)
    _write_jsonl(rag_answers_path, rag_results)

    direct_source = "local"
    rag_summary: Dict[str, object] | None = None
    direct_summary: Dict[str, object] | None = None
    fallback_warning: str | None = None

    try:
        if eval_mode == "ragas":
            direct_results = _build_direct_results_openai(rows, direct_model, persist_path=direct_answers_path)
            direct_source = "model"
            _write_jsonl(direct_answers_path, direct_results)
            rag_df, rag_summary = _run_ragas_subprocess(rag_answers_path, out_dir, "rag", settings.llm_model)
            direct_df, direct_summary = _run_ragas_subprocess(direct_answers_path, out_dir, "direct", direct_model)
            rag_df = _attach_metadata(rag_df, rag_results)
            direct_df = _attach_metadata(direct_df, direct_results)
        else:
            direct_results = _build_direct_results_local(rows)
            rag_df = evaluate_local_results(rag_results)
            direct_df = evaluate_local_results(direct_results)
    except Exception as exc:
        if eval_mode_requested == "ragas":
            raise
        eval_mode = "local"
        try:
            direct_results = _build_direct_results_openai(rows, direct_model, persist_path=direct_answers_path)
            direct_source = "model"
            notice = build_fallback_notice(exc, None)
        except Exception as direct_exc:
            direct_results = _build_direct_results_local(rows)
            direct_source = "local"
            notice = build_fallback_notice(exc, direct_exc)
        mode_reason = notice["mode_reason"]
        fallback_warning = notice["warning"]
        if fail_on_fallback:
            raise RuntimeError(fallback_warning) from exc
        print(f"WARNING: {fallback_warning}", file=sys.stderr)
        rag_df = evaluate_local_results(rag_results)
        direct_df = evaluate_local_results(direct_results)

    _write_jsonl(direct_answers_path, direct_results)

    if eval_mode == "ragas":
        assert rag_summary is not None
        assert direct_summary is not None
        rag_summary.update({k: v for k, v in _operational_summary(rag_results, "RAG").items() if k not in {"method", "rows"}})
        direct_summary.update({k: v for k, v in _operational_summary(direct_results, "Direct").items() if k not in {"method", "rows"}})
        rag_summary["method"] = "RAG"
        rag_summary["eval_mode"] = eval_mode
        direct_summary["method"] = "Direct"
        direct_summary["eval_mode"] = eval_mode
        direct_summary["direct_source"] = direct_source
        direct_summary["model"] = direct_model
    else:
        rag_summary = compute_metrics(
            rag_df,
            method="RAG",
            extra={"eval_mode": eval_mode},
        )
        direct_extra = {"eval_mode": eval_mode, "direct_source": direct_source}
        if direct_source == "model":
            direct_extra["model"] = direct_model
        direct_summary = compute_metrics(
            direct_df,
            method="Direct",
            extra=direct_extra,
        )

    comparison_df = _build_comparison_frame(eval_mode, rag_summary, direct_summary)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(frame_to_markdown(comparison_df), encoding="utf-8")
    _write_metric_plot(comparison_df, out_dir / "comparison_summary.png", title="RAG vs Direct Comparison")
    category_breakdown = build_category_breakdown(
        pd.concat([rag_df.assign(method="RAG"), direct_df.assign(method="Direct")], ignore_index=True)
    )
    category_breakdown.to_csv(out_dir / "category_breakdown.csv", index=False)
    (out_dir / "category_breakdown.md").write_text(frame_to_markdown(category_breakdown), encoding="utf-8")

    run_meta = {
        "data_path": str(data_path_label),
        "samples": len(rows),
        "eval_mode": eval_mode,
    }
    if mode_reason:
        run_meta["mode_reason"] = mode_reason
    if fallback_warning:
        run_meta["warning"] = fallback_warning

    (out_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Eval Summary", run_meta),
                "## Comparison\n\n" + frame_to_markdown(comparison_df),
                "## Category Breakdown\n\n" + frame_to_markdown(category_breakdown),
                metrics_to_markdown("## RAG Metrics", rag_summary),
                metrics_to_markdown("## Direct Metrics", direct_summary),
            ]
        ),
        encoding="utf-8",
    )
    summary_payload = {
        "data_path": str(data_path_label),
        "samples": len(rows),
        "eval_mode": eval_mode,
        "top_k": top_k,
        "candidate_top_k": _candidate_top_k_value(settings),
        "mode_reason": mode_reason,
        "warning": fallback_warning,
        "rag": rag_summary,
        "direct": direct_summary,
        "category_breakdown_path": str(out_dir / "category_breakdown.csv"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


def _round_record(
    method_summary: Dict[str, object],
    round_index: int,
    round_seed: int,
    sample_rows: Sequence[Dict],
    model_label: str,
    model_name: str,
) -> Dict[str, object]:
    record = dict(method_summary)
    record["model_label"] = model_label
    record["model"] = model_name
    record["round"] = round_index
    record["seed"] = round_seed
    record["sample_size"] = len(sample_rows)
    record["unique_questions"] = len({str(row.get("question", "")) for row in sample_rows})
    return record


def _aggregate_sampling_rounds(round_records: Sequence[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(round_records)
    if frame.empty:
        return pd.DataFrame()
    group_cols = ["method"]
    if "model_label" in frame.columns:
        group_cols.insert(0, "model_label")
    if "model" in frame.columns:
        group_cols.insert(1, "model")
    return summarize_grouped_distribution(frame, group_cols=group_cols, exclude_numeric_cols={"round", "seed", "rows"})


def _primary_metric_column(frame: pd.DataFrame) -> str | None:
    for candidate in (
        "answer_relevancy_mean",
        "answer_token_f1_mean",
        "keyword_hit_rate_mean",
        "source_hit_rate_mean",
    ):
        if candidate in frame.columns:
            return candidate
    return None


def _build_leaderboard(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    metric_column = _primary_metric_column(summary_frame)
    leaderboard = summary_frame.copy()
    leaderboard["primary_metric"] = metric_column or ""
    if metric_column:
        leaderboard["primary_score"] = leaderboard[metric_column]
        sort_cols = ["primary_score"]
        ascending = [False]
        if "rows" in leaderboard.columns:
            sort_cols.append("rows")
            ascending.append(False)
        for optional in ("model_label", "method"):
            if optional in leaderboard.columns:
                sort_cols.append(optional)
                ascending.append(True)
        leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        leaderboard["rank"] = range(1, len(leaderboard) + 1)
    else:
        leaderboard["primary_score"] = None
        leaderboard["rank"] = range(1, len(leaderboard) + 1)
    columns = ["rank", "model_label", "model", "method", "primary_metric", "primary_score", "rows"]
    available = [column for column in columns if column in leaderboard.columns]
    return leaderboard[available]


def _write_frame_artifacts(frame: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(frame_to_markdown(frame), encoding="utf-8")


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _run_compare_sampling(
    rows: Sequence[Dict],
    out_dir: Path,
    settings: Settings,
    top_k: int,
    eval_mode_requested: str,
    direct_model: str,
    fail_on_fallback: bool,
    data_path_label: str,
    rounds: int,
    sample_size: int,
    with_replacement: bool,
    seed: int,
    model_label: str | None = None,
    model_name: str | None = None,
) -> Dict[str, object]:
    resolved_model_label = model_label or settings.llm_model
    resolved_model_name = model_name or settings.llm_model
    round_records: list[Dict[str, object]] = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + round_index - 1
        sampled_rows = _sample_rows(rows, sample_size=sample_size, with_replacement=with_replacement, seed=round_seed)
        round_dir = out_dir / f"round_{round_index:03d}"
        summary = _run_compare_once(
            sampled_rows,
            round_dir,
            settings=settings,
            top_k=top_k,
            eval_mode_requested=eval_mode_requested,
            direct_model=direct_model,
            fail_on_fallback=fail_on_fallback,
            data_path_label=f"{data_path_label}#round_{round_index:03d}",
        )
        round_records.append(
            _round_record(
                summary["rag"],
                round_index,
                round_seed,
                sampled_rows,
                model_label=resolved_model_label,
                model_name=resolved_model_name,
            )
        )
        round_records.append(
            _round_record(
                summary["direct"],
                round_index,
                round_seed,
                sampled_rows,
                model_label=resolved_model_label,
                model_name=resolved_model_name,
            )
        )

    rounds_df = pd.DataFrame(round_records)
    _write_frame_artifacts(rounds_df, out_dir / "per_round_results.csv", out_dir / "per_round_results.md")
    _write_frame_artifacts(rounds_df, out_dir / "sampling_rounds.csv", out_dir / "sampling_rounds.md")

    aggregate_df = _aggregate_sampling_rounds(round_records)
    _write_frame_artifacts(aggregate_df, out_dir / "model_summary.csv", out_dir / "model_summary.md")
    _write_frame_artifacts(aggregate_df, out_dir / "comparison.csv", out_dir / "comparison.md")
    _write_metric_plot(aggregate_df, out_dir / "comparison_summary.png", title="Repeated-Sampling RAG vs Direct Comparison")
    leaderboard_df = _build_leaderboard(aggregate_df)
    _write_frame_artifacts(leaderboard_df, out_dir / "leaderboard.csv", out_dir / "leaderboard.md")

    summary_payload = {
        "data_path": str(data_path_label),
        "model_label": resolved_model_label,
        "model": resolved_model_name,
        "population_size": len(rows),
        "rounds": rounds,
        "sample_size": sample_size,
        "with_replacement": with_replacement,
        "seed": seed,
        "top_k": top_k,
        "candidate_top_k": _candidate_top_k_value(settings),
        "comparison_path": str(out_dir / "comparison.csv"),
        "model_summary_path": str(out_dir / "model_summary.csv"),
        "leaderboard_path": str(out_dir / "leaderboard.csv"),
        "per_round_results_path": str(out_dir / "per_round_results.csv"),
        "rounds_path": str(out_dir / "sampling_rounds.csv"),
        "round_records": round_records,
    }
    summary_file_payload = {key: value for key, value in summary_payload.items() if key != "round_records"}
    (out_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Sampling Eval Summary", summary_file_payload),
                "## Aggregate Comparison\n\n" + frame_to_markdown(aggregate_df),
                "## Leaderboard\n\n" + frame_to_markdown(leaderboard_df),
                "## Round-Level Records\n\n" + frame_to_markdown(rounds_df),
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary_file_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def _run_compare_multimodel_sampling(
    rows: Sequence[Dict],
    out_dir: Path,
    settings: Settings,
    top_k: int,
    eval_mode_requested: str,
    fail_on_fallback: bool,
    data_path_label: str,
    rounds: int,
    sample_size: int,
    with_replacement: bool,
    seed: int,
    benchmark_models: Sequence[Dict[str, str]],
    candidate_top_k: int | None = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "data_path": str(data_path_label),
        "population_size": len(rows),
        "rounds": rounds,
        "sample_size": sample_size,
        "with_replacement": with_replacement,
        "seed": seed,
        "top_k": top_k,
        "candidate_top_k": candidate_top_k or _candidate_top_k_value(settings),
        "benchmark_models": list(benchmark_models),
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_model_summaries: list[Dict[str, object]] = []
    all_round_records: list[Dict[str, object]] = []
    model_summary_frames: list[pd.DataFrame] = []

    for spec in benchmark_models:
        model_dir = out_dir / "models" / spec["slug"]
        model_settings = _settings_with_candidate_top_k(
            _settings_with_model(settings, spec["model"]),
            candidate_top_k,
        )
        summary = _run_compare_sampling(
            rows,
            model_dir,
            settings=model_settings,
            top_k=top_k,
            eval_mode_requested=eval_mode_requested,
            direct_model=spec["model"],
            fail_on_fallback=fail_on_fallback,
            data_path_label=data_path_label,
            rounds=rounds,
            sample_size=sample_size,
            with_replacement=with_replacement,
            seed=seed,
            model_label=spec["label"],
            model_name=spec["model"],
        )
        per_model_summaries.append(
            {
                "label": spec["label"],
                "model": spec["model"],
                "slug": spec["slug"],
                "summary_path": str(model_dir / "summary.json"),
            }
        )
        all_round_records.extend(summary.get("round_records", []))
        model_summary_frames.append(_load_frame(model_dir / "model_summary.csv"))

    per_round_df = pd.DataFrame(all_round_records)
    _write_frame_artifacts(per_round_df, out_dir / "per_round_results.csv", out_dir / "per_round_results.md")
    _write_frame_artifacts(per_round_df, out_dir / "sampling_rounds.csv", out_dir / "sampling_rounds.md")

    if model_summary_frames:
        aggregate_df = pd.concat(model_summary_frames, ignore_index=True)
    else:
        aggregate_df = _aggregate_sampling_rounds(all_round_records)
    _write_frame_artifacts(aggregate_df, out_dir / "model_summary.csv", out_dir / "model_summary.md")
    _write_frame_artifacts(aggregate_df, out_dir / "comparison.csv", out_dir / "comparison.md")
    _write_metric_plot(aggregate_df, out_dir / "comparison_summary.png", title="Multi-Model RAG vs Direct Comparison")

    leaderboard_df = _build_leaderboard(aggregate_df)
    _write_frame_artifacts(leaderboard_df, out_dir / "leaderboard.csv", out_dir / "leaderboard.md")

    summary_payload = {
        "data_path": str(data_path_label),
        "population_size": len(rows),
        "model_count": len(benchmark_models),
        "rounds": rounds,
        "sample_size": sample_size,
        "with_replacement": with_replacement,
        "seed": seed,
        "top_k": top_k,
        "candidate_top_k": candidate_top_k or _candidate_top_k_value(settings),
        "experiment_config_path": str(out_dir / "experiment_config.json"),
        "per_round_results_path": str(out_dir / "per_round_results.csv"),
        "model_summary_path": str(out_dir / "model_summary.csv"),
        "leaderboard_path": str(out_dir / "leaderboard.csv"),
        "models": per_model_summaries,
    }
    (out_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Multi-Model Sampling Summary", summary_payload),
                "## Leaderboard\n\n" + frame_to_markdown(leaderboard_df),
                "## Model Summary\n\n" + frame_to_markdown(aggregate_df),
                "## Round-Level Records\n\n" + frame_to_markdown(per_round_df),
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.compare")
    parser.add_argument("--data-path", default="data/fbtp_eval.jsonl")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--model", default=None, help="Direct-answering model. Defaults to Settings().llm_model.")
    parser.add_argument(
        "--benchmark-model",
        action="append",
        default=[],
        help="Benchmark model spec. Use plain model name or label=model_name. Repeat for multiple models.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        help="Retrieval candidate pool size before final top-k truncation. Defaults to the configured reranker_top_n.",
    )
    parser.add_argument("--eval-mode", choices=["auto", "ragas", "local"], default="auto")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--rounds", type=int, default=1, help="Number of repeated benchmark rounds to run.")
    parser.add_argument("--sample-size", type=int, default=None, help="Rows to sample in each round. Defaults to full dataset size.")
    parser.add_argument("--with-replacement", action="store_true", help="Sample with replacement for repeated benchmark runs.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for repeated sampling.")
    args = parser.parse_args()

    load_dotenv()
    settings = Settings()
    settings = _settings_with_candidate_top_k(settings, args.candidate_top_k)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    rows = _load_jsonl(data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    direct_model = args.model or settings.llm_model
    benchmark_models = _parse_benchmark_model_specs(args.benchmark_model, default_model=settings.llm_model)
    rounds = max(1, args.rounds)
    sample_size = args.sample_size or len(rows)
    if benchmark_models:
        _run_compare_multimodel_sampling(
            rows,
            out_dir,
            settings=settings,
            top_k=args.top_k,
            candidate_top_k=args.candidate_top_k,
            eval_mode_requested=args.eval_mode,
            fail_on_fallback=args.fail_on_fallback,
            data_path_label=str(data_path),
            rounds=rounds,
            sample_size=sample_size,
            with_replacement=args.with_replacement,
            seed=args.seed,
            benchmark_models=benchmark_models,
        )
    elif rounds > 1 or args.sample_size is not None:
        _run_compare_sampling(
            rows,
            out_dir,
            settings=settings,
            top_k=args.top_k,
            eval_mode_requested=args.eval_mode,
            direct_model=direct_model,
            fail_on_fallback=args.fail_on_fallback,
            data_path_label=str(data_path),
            rounds=rounds,
            sample_size=sample_size,
            with_replacement=args.with_replacement,
            seed=args.seed,
            model_label=settings.llm_model,
            model_name=settings.llm_model,
        )
    else:
        _run_compare_once(
            rows,
            out_dir,
            settings=settings,
            top_k=args.top_k,
            eval_mode_requested=args.eval_mode,
            direct_model=direct_model,
            fail_on_fallback=args.fail_on_fallback,
            data_path_label=str(data_path),
        )

    print("Saved comparison to", out_dir)


if __name__ == "__main__":
    main()
