from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipelines.schema_benchmark_dataset import build_schema_benchmark_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = REPO_ROOT.parent
DEFAULT_RAGKB_ROOT = PROJECTS_ROOT / "llm-rag-knowledge-base"
DEFAULT_RAGPPI_PATH = DEFAULT_RAGKB_ROOT / "data" / "hf_ragppi_sample" / "ragppi_eval.jsonl"
DEFAULT_DOC_DESIGN_PATH = REPO_ROOT / "data" / "doc_design_eval_20.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "fbtp_eval_fixed_120.jsonl"
DEFAULT_SCHEMA_COUNTS = {
    "protein_profile": 8,
    "protein_identifier": 4,
    "interaction_overview": 8,
    "affinity": 8,
    "developability": 6,
    "annotation": 4,
    "provenance": 4,
    "structure": 4,
    "digestive_assay": 2,
    "immunogenicity": 2,
    "loop_annotation": 2,
    "loop_flexibility": 2,
    "protein_flexibility": 2,
    "target_variant": 2,
    "source_metadata": 2,
}


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    text = str(value).strip()
    return [text] if text else []


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_schema_dir(schema_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    if schema_dir is not None:
        candidates.append(Path(schema_dir))
    env_value = os.getenv("SCHEMA_TABLES_DIR")
    if env_value:
        candidates.append(Path(env_value))

    summary_path = DEFAULT_RAGKB_ROOT / "data" / "schema_tables_rag_ready" / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            resolved = summary.get("schema_dir")
            if resolved:
                resolved_text = str(resolved)
                if "<local_path_removed>" not in resolved_text:
                    candidates.append(Path(resolved_text))
        except json.JSONDecodeError:
            pass

    candidates.append(Path(r"<local_path_removed>"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve schema tables directory. Pass --schema-dir or set SCHEMA_TABLES_DIR.")


def _normalize_row(row: dict[str, Any], source_group: str, source_path: Path) -> dict[str, Any] | None:
    question = str(row.get("question", "")).strip()
    ground_truth = str(row.get("ground_truth", "")).strip()
    if not question or not ground_truth:
        return None

    tags = _normalize_list(row.get("tags"))
    normalized = {
        "question": question,
        "record_type": str(row.get("record_type") or "unknown").strip() or "unknown",
        "ground_truth": ground_truth,
        "expected_answer_contains": _normalize_list(row.get("expected_answer_contains")),
        "expected_source_contains": _normalize_list(row.get("expected_source_contains")),
        "expected_table_contains": _normalize_list(row.get("expected_table_contains")),
        "expected_primary_ids": _normalize_list(row.get("expected_primary_ids")),
        "tags": list(dict.fromkeys(tags + ["benchmark_fixed_set", source_group])),
        "benchmark_source_group": source_group,
        "benchmark_source_path": str(source_path),
    }
    if "query_type" in row and row.get("query_type"):
        normalized["query_type"] = row.get("query_type")
    if "difficulty" in row and row.get("difficulty"):
        normalized["difficulty"] = row.get("difficulty")
    return normalized


def _dedupe_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["question"].strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def build_fixed_benchmark_rows(
    ragppi_path: Path | None = None,
    doc_design_path: Path | None = None,
    schema_dir: Path | None = None,
    ragppi_count: int = 40,
    doc_count: int = 20,
    schema_counts: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    ragppi_path = Path(ragppi_path or DEFAULT_RAGPPI_PATH)
    doc_design_path = Path(doc_design_path or DEFAULT_DOC_DESIGN_PATH)
    resolved_schema_dir = _resolve_schema_dir(schema_dir)

    ragppi_rows = _load_jsonl(ragppi_path)[:ragppi_count]
    doc_rows = _load_jsonl(doc_design_path)[:doc_count]
    schema_rows = build_schema_benchmark_rows(resolved_schema_dir, counts=schema_counts or DEFAULT_SCHEMA_COUNTS, seed=seed)

    merged: list[dict[str, Any]] = []
    for row in ragppi_rows:
        normalized = _normalize_row(row, "ragppi_gold", ragppi_path)
        if normalized is not None:
            merged.append(normalized)
    for row in doc_rows:
        normalized = _normalize_row(row, "doc_design", doc_design_path)
        if normalized is not None:
            merged.append(normalized)
    for row in schema_rows:
        normalized = _normalize_row(row, "schema_tables", resolved_schema_dir)
        if normalized is not None:
            merged.append(normalized)

    deduped = _dedupe_rows(merged)
    for idx, row in enumerate(deduped, start=1):
        row["question_id"] = f"fixed-{idx:03d}"
    return deduped


def _count_values(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(counter)


def write_fixed_benchmark_dataset(
    output_path: Path | None = None,
    ragppi_path: Path | None = None,
    doc_design_path: Path | None = None,
    schema_dir: Path | None = None,
    ragppi_count: int = 40,
    doc_count: int = 20,
    schema_counts: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
    rows = build_fixed_benchmark_rows(
        ragppi_path=ragppi_path,
        doc_design_path=doc_design_path,
        schema_dir=schema_dir,
        ragppi_count=ragppi_count,
        doc_count=doc_count,
        schema_counts=schema_counts,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = {
        "output_path": str(output_path),
        "rows": len(rows),
        "ragppi_count": ragppi_count,
        "doc_count": doc_count,
        "schema_counts": schema_counts or DEFAULT_SCHEMA_COUNTS,
        "source_group_counts": _count_values(rows, "benchmark_source_group"),
        "record_type_counts": _count_values(rows, "record_type"),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.prepare_benchmark_dataset")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--ragppi-path", default=str(DEFAULT_RAGPPI_PATH))
    parser.add_argument("--doc-design-path", default=str(DEFAULT_DOC_DESIGN_PATH))
    parser.add_argument("--schema-dir", default=None)
    parser.add_argument("--ragppi-count", type=int, default=40)
    parser.add_argument("--doc-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = write_fixed_benchmark_dataset(
        output_path=Path(args.output_path),
        ragppi_path=Path(args.ragppi_path),
        doc_design_path=Path(args.doc_design_path),
        schema_dir=Path(args.schema_dir) if args.schema_dir else None,
        ragppi_count=args.ragppi_count,
        doc_count=args.doc_count,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
