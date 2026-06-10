from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable

import pandas as pd


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _coerce_frame(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, dict):
        return pd.DataFrame([results])
    if isinstance(results, Iterable) and not isinstance(results, (str, bytes)):
        return pd.DataFrame(list(results))
    raise TypeError("results must be a pandas DataFrame, mapping, or iterable of mappings")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "/")


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available."

    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_stringify(cell) for cell in row) + " |")
    return "\n".join(lines)


def compute_metrics(
    results: Any,
    method: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = _coerce_frame(results)
    numeric = df.select_dtypes(include="number")

    summary: dict[str, Any] = {}
    if method:
        summary["method"] = method
    summary["rows"] = int(len(df))

    if not numeric.empty:
        for col, value in numeric.mean().items():
            if pd.notna(value):
                summary[col] = round(float(value), 4)

    if extra:
        summary.update(extra)
    return summary


def summarize_grouped_distribution(
    results: Any,
    group_cols: list[str],
    exclude_numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = _coerce_frame(results)
    if df.empty:
        return pd.DataFrame()

    valid_group_cols = [column for column in group_cols if column in df.columns]
    if not valid_group_cols:
        raise ValueError("At least one group column must exist in the results")

    excluded = set(exclude_numeric_cols or [])
    numeric_cols = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(valid_group_cols, dropna=False)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = {
            column: value
            for column, value in zip(valid_group_cols, group_key)
        }
        record["rows"] = int(len(group_frame))
        for column in numeric_cols:
            series = group_frame[column].dropna()
            if series.empty:
                continue
            record[f"{column}_mean"] = round(float(series.mean()), 4)
            record[f"{column}_std"] = round(float(series.std(ddof=1)), 4) if len(series) > 1 else 0.0
            record[f"{column}_min"] = round(float(series.min()), 4)
            record[f"{column}_max"] = round(float(series.max()), 4)
            record[f"{column}_p05"] = round(float(series.quantile(0.05)), 4)
            record[f"{column}_p50"] = round(float(series.quantile(0.50)), 4)
            record[f"{column}_p95"] = round(float(series.quantile(0.95)), 4)
        rows.append(record)

    summary = pd.DataFrame(rows)
    sort_cols = ["rows"] + valid_group_cols
    ascending = [False] + [True] * len(valid_group_cols)
    return summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def metrics_to_markdown(title: str, metrics: dict[str, Any]) -> str:
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    table = frame_to_markdown(pd.DataFrame(rows)) if rows else "No metrics available."
    return f"{title}\n\n{table}"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def estimate_row_cost(
    row: dict[str, Any],
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    prompt_tokens = float(row.get("prompt_tokens", 0) or 0)
    completion_tokens = float(row.get("completion_tokens", 0) or 0)
    input_cost = (prompt_tokens / 1_000_000.0) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000.0) * output_cost_per_million
    return round(input_cost + output_cost, 6)


def build_category_breakdown(results: Any) -> pd.DataFrame:
    df = _coerce_frame(results)
    if df.empty or "category" not in df.columns:
        return pd.DataFrame(columns=["category", "rows"])
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    group_cols = ["category"]
    if "method" in df.columns:
        group_cols.insert(0, "method")
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped[numeric_cols].mean(numeric_only=True).reset_index() if numeric_cols else grouped.size().reset_index(name="rows")
    counts = grouped.size().reset_index(name="rows")
    if "rows" not in summary.columns:
        summary = summary.merge(counts, on=group_cols, how="left")
    sort_cols = ["rows"] + group_cols
    ascending = [False] + [True] * len(group_cols)
    return summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _token_f1(answer: str, ground_truth: str) -> tuple[float, float, float]:
    answer_tokens = _tokenize(answer)
    truth_tokens = _tokenize(ground_truth)
    if not answer_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0

    answer_counts = Counter(answer_tokens)
    truth_counts = Counter(truth_tokens)
    overlap = sum((answer_counts & truth_counts).values())
    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(truth_tokens) if truth_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _hit_rate(text: str, expected_values: Any) -> float:
    expected = _coerce_list(expected_values)
    if not expected:
        return 0.0
    haystack = (text or "").lower()
    hits = sum(1 for item in expected if item.lower() in haystack)
    return hits / len(expected)


def list_hit_rate(actual_values: Any, expected_values: Any) -> float:
    actual = {item.lower() for item in _coerce_list(actual_values)}
    expected = _coerce_list(expected_values)
    if not expected:
        return 0.0
    hits = sum(1 for item in expected if item.lower() in actual)
    return hits / len(expected)


def evaluate_local_results(results: Any) -> pd.DataFrame:
    df = _coerce_frame(results)
    rows = []
    for row in df.to_dict(orient="records"):
        answer = row.get("answer", "") or ""
        ground_truth = row.get("ground_truth", "") or ""
        source_markers = " ".join(_coerce_list(row.get("source_markers")))
        table_markers = row.get("table_markers", [])
        id_markers = row.get("id_markers", [])
        raw_scores = row.get("context_scores") or []
        if isinstance(raw_scores, str):
            raw_scores = [raw_scores]
        context_scores = []
        for score in raw_scores:
            if score in (None, ""):
                continue
            try:
                context_scores.append(float(score))
            except (TypeError, ValueError):
                continue
        precision, recall, f1 = _token_f1(answer, ground_truth)

        rows.append(
            {
                "question": row.get("question", ""),
                "category": row.get("category", row.get("record_type", "unknown")),
                "answer_token_precision": round(precision, 4),
                "answer_token_recall": round(recall, 4),
                "answer_token_f1": round(f1, 4),
                "keyword_hit_rate": round(_hit_rate(answer, row.get("expected_answer_contains")), 4),
                "source_hit_rate": round(_hit_rate(source_markers, row.get("expected_source_contains")), 4),
                "table_hit_rate": round(list_hit_rate(table_markers, row.get("expected_table_contains")), 4),
                "id_hit_rate": round(list_hit_rate(id_markers, row.get("expected_primary_ids")), 4),
                "context_count": int(row.get("context_count", 0) or 0),
                "answer_chars": len(answer),
                "max_context_score": round(max(context_scores), 4) if context_scores else 0.0,
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
                "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0) or 0.0),
            }
        )

    return pd.DataFrame(rows)
