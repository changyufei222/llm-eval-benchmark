from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


DOC_DESIGN_BUCKETS = {
    "doc_design",
    "architecture",
    "quality",
    "database",
    "provenance",
    "benchmark_protocol",
    "doc_methodology",
}

SCHEMA_TABLE_BUCKETS = {
    "schema_tables",
    "protein_profile",
    "protein_identifier",
    "interaction_overview",
    "affinity",
    "developability",
    "annotation",
    "structure",
    "digestive_assay",
    "immunogenicity",
    "loop_annotation",
    "loop_flexibility",
    "protein_flexibility",
    "target_variant",
    "source_metadata",
}


def _coerce_frame(results: object) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def normalize_category_bucket(category: object) -> str:
    value = str(category or "").strip()
    if not value:
        return "unknown"
    lowered = value.lower()
    if lowered == "ragppi":
        return "ragppi"
    if lowered in DOC_DESIGN_BUCKETS:
        return "doc/design"
    if lowered in SCHEMA_TABLE_BUCKETS:
        return "schema_tables"
    return value


def _ci95_bounds(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return (float("nan"), float("nan"))
    mean = float(clean.mean())
    if len(clean) == 1:
        return (round(mean, 4), round(mean, 4))
    std = float(clean.std(ddof=1))
    half_width = 1.96 * std / math.sqrt(len(clean))
    return (round(mean - half_width, 4), round(mean + half_width, 4))


def summarize_benchmark_distribution(
    results: object,
    group_cols: list[str],
    exclude_numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    frame = _coerce_frame(results)
    if frame.empty:
        return pd.DataFrame()

    valid_group_cols = [column for column in group_cols if column in frame.columns]
    if not valid_group_cols:
        raise ValueError("At least one group column must exist in the results")

    excluded = set(exclude_numeric_cols or [])
    numeric_cols = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded
    ]

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(valid_group_cols, dropna=False)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = {column: value for column, value in zip(valid_group_cols, group_key)}
        record["rows"] = int(len(group_frame))
        for column in numeric_cols:
            series = group_frame[column].dropna()
            if series.empty:
                continue
            ci_low, ci_high = _ci95_bounds(series)
            record[f"{column}_mean"] = round(float(series.mean()), 4)
            record[f"{column}_std"] = round(float(series.std(ddof=1)), 4) if len(series) > 1 else 0.0
            record[f"{column}_p50"] = round(float(series.quantile(0.50)), 4)
            record[f"{column}_p95"] = round(float(series.quantile(0.95)), 4)
            record[f"{column}_ci95_low"] = ci_low
            record[f"{column}_ci95_high"] = ci_high
        rows.append(record)

    summary = pd.DataFrame(rows)
    sort_cols = ["rows"] + valid_group_cols
    ascending = [False] + [True] * len(valid_group_cols)
    return summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def summarize_rag_direct_uplift(results: object) -> pd.DataFrame:
    frame = _coerce_frame(results)
    if frame.empty or "method" not in frame.columns:
        return pd.DataFrame()

    identity_cols = [
        column
        for column in (
            "model_label",
            "model",
            "group_id",
            "global_round",
            "round",
            "seed",
            "sample_size",
            "unique_questions",
        )
        if column in frame.columns
    ]
    numeric_excluded = {"round", "seed", "sample_size", "unique_questions", "rows"}
    metric_cols = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in numeric_excluded and column not in identity_cols
    ]
    rag = frame[frame["method"] == "RAG"]
    direct = frame[frame["method"] == "Direct"]
    if rag.empty or direct.empty or not metric_cols:
        return pd.DataFrame()

    merged = rag[identity_cols + metric_cols].merge(
        direct[identity_cols + metric_cols],
        on=identity_cols,
        how="inner",
        suffixes=("_rag", "_direct"),
    )
    if merged.empty:
        return pd.DataFrame()

    delta_records: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        record = {column: row[column] for column in identity_cols}
        record["comparison"] = "RAG - Direct"
        for column in metric_cols:
            rag_value = row.get(f"{column}_rag")
            direct_value = row.get(f"{column}_direct")
            if pd.notna(rag_value) and pd.notna(direct_value):
                record[column] = float(rag_value) - float(direct_value)
        delta_records.append(record)

    delta_frame = pd.DataFrame(delta_records)
    group_cols = [column for column in ("model_label", "model", "comparison") if column in delta_frame.columns]
    if not group_cols:
        group_cols = ["comparison"]
    return summarize_benchmark_distribution(
        delta_frame,
        group_cols=group_cols,
        exclude_numeric_cols={"round", "seed", "sample_size", "unique_questions", "global_round", "group_index"},
    )


def _load_benchmark_model_map(experiment_config_path: Path) -> dict[str, dict[str, str]]:
    if not experiment_config_path.exists():
        return {}
    payload = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    model_map: dict[str, dict[str, str]] = {}
    for spec in payload.get("benchmark_models", []):
        slug = str(spec.get("slug", "")).strip()
        if slug:
            model_map[slug] = {
                "model_label": str(spec.get("label", slug)),
                "model": str(spec.get("model", spec.get("label", slug))),
            }
    return model_map


def combine_main_category_breakdowns(main_dir: Path | str) -> pd.DataFrame:
    root = Path(main_dir)
    model_map = _load_benchmark_model_map(root / "experiment_config.json")
    csv_paths = sorted(root.glob("models/*/round_001/category_breakdown.csv"))
    if not csv_paths:
        return pd.DataFrame()

    combined_rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        slug = csv_path.parents[1].name
        meta = model_map.get(slug, {"model_label": slug, "model": slug})
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            record = dict(row)
            record.update(meta)
            record["model_slug"] = slug
            record["category_bucket"] = normalize_category_bucket(row.get("category"))
            combined_rows.append(record)

    combined = pd.DataFrame(combined_rows)
    if combined.empty:
        return combined

    group_cols = ["model_label", "model", "method", "category_bucket"]
    numeric_cols = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column != "rows"
    ]

    rows: list[dict[str, object]] = []
    grouped = combined.groupby(group_cols, dropna=False)
    for group_key, group_frame in grouped:
        record = {column: value for column, value in zip(group_cols, group_key)}
        total_rows = int(group_frame["rows"].fillna(0).sum())
        record["rows"] = total_rows
        weights = group_frame["rows"].fillna(0).astype(float)
        for column in numeric_cols:
            series = group_frame[column]
            valid_mask = series.notna() & weights.gt(0)
            if not valid_mask.any():
                continue
            weighted_values = series[valid_mask].astype(float)
            weighted_weights = weights[valid_mask]
            denominator = float(weighted_weights.sum())
            if denominator <= 0:
                continue
            record[column] = round(float((weighted_values * weighted_weights).sum() / denominator), 4)
        rows.append(record)

    summary = pd.DataFrame(rows)
    return summary.sort_values(["model_label", "method", "category_bucket"], ascending=[True, True, True]).reset_index(drop=True)


def pivot_method_metrics(
    results: object,
    *,
    index_cols: list[str],
    value_cols: list[str],
    method_col: str = "method",
    method_values: tuple[str, ...] = ("Direct", "RAG"),
) -> pd.DataFrame:
    frame = _coerce_frame(results)
    if frame.empty:
        columns = list(index_cols)
        for value_col in value_cols:
            for method in method_values:
                columns.append(f"{value_col}_{method.lower()}")
        return pd.DataFrame(columns=columns)

    valid_index_cols = [column for column in index_cols if column in frame.columns]
    if not valid_index_cols:
        raise ValueError("At least one index column must exist in the results")
    if method_col not in frame.columns:
        raise ValueError(f"{method_col} column must exist in the results")

    available_value_cols = [column for column in value_cols if column in frame.columns]
    base = frame[valid_index_cols].drop_duplicates().reset_index(drop=True)
    if not available_value_cols:
        for value_col in value_cols:
            for method in method_values:
                base[f"{value_col}_{method.lower()}"] = float("nan")
        return base

    pivot_source = (
        frame[valid_index_cols + [method_col] + available_value_cols]
        .groupby(valid_index_cols + [method_col], dropna=False, as_index=True)
        .first()
    )
    pivoted = pivot_source.unstack(method_col).reset_index()

    flattened: list[str] = []
    for column in pivoted.columns:
        if isinstance(column, tuple):
            left, right = column
            if right in ("", None):
                flattened.append(str(left))
            else:
                flattened.append(f"{left}_{str(right).lower()}")
        else:
            flattened.append(str(column))
    pivoted.columns = flattened

    for value_col in value_cols:
        for method in method_values:
            column_name = f"{value_col}_{method.lower()}"
            if column_name not in pivoted.columns:
                pivoted[column_name] = float("nan")

    ordered_cols = list(valid_index_cols)
    for value_col in value_cols:
        for method in method_values:
            ordered_cols.append(f"{value_col}_{method.lower()}")

    return pivoted[ordered_cols].sort_values(valid_index_cols).reset_index(drop=True)
