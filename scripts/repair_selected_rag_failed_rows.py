from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.metrics import compute_metrics, frame_to_markdown, metrics_to_markdown
from metrics.plotting import write_metric_plot
from pipelines import compare
from pipelines.benchmark_protocol import FINAL_CONTEXT_TOP_K


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_target(value: str) -> tuple[str, str]:
    cleaned = value.strip().replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) != 2 or not parts[0].startswith("group_") or not parts[1].startswith("round_"):
        raise argparse.ArgumentTypeError(f"invalid target '{value}', expected group_XX/round_YYY")
    return parts[0], parts[1]


def _round_index(round_name: str) -> int:
    return int(round_name.split("_")[-1])


def _sample_rows_from_direct_answers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = compare._load_jsonl(path)
    return [
        {
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", ""),
            "record_type": row.get("record_type"),
            "tags": row.get("tags", []),
            "expected_answer_contains": row.get("expected_answer_contains", []),
            "expected_source_contains": row.get("expected_source_contains", []),
            "expected_table_contains": row.get("expected_table_contains", []),
            "expected_primary_ids": row.get("expected_primary_ids", []),
        }
        for row in rows
    ]


def _load_group_round_seed_map(group_dir: Path) -> dict[int, int]:
    csv_path = group_dir / "per_round_results.csv"
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    if frame.empty or "round" not in frame.columns or "seed" not in frame.columns:
        return {}
    mapping: dict[int, int] = {}
    for row in frame.to_dict(orient="records"):
        try:
            round_index = int(row["round"])
            mapping.setdefault(round_index, int(row["seed"]))
        except Exception:
            continue
    return mapping


def _load_existing_rag_summary(round_dir: Path) -> dict[str, Any] | None:
    summary_path = round_dir / "summary.json"
    if not summary_path.exists():
        return None
    payload = _load_json(summary_path)
    if "rag" in payload and isinstance(payload["rag"], dict):
        return dict(payload["rag"])
    if str(payload.get("method", "")).upper() == "RAG":
        return dict(payload)
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _run_ragas_subprocess_with_timeout(
    answers_path: Path,
    out_dir: Path,
    metric_set: str,
    model: str,
    *,
    timeout_seconds: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        timeout=timeout_seconds,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())

    df = pd.read_csv(out_dir / f"{scores_name}.csv")
    summary = _load_json(out_dir / f"{summary_name}.json")
    return df, summary


def _write_ragas_summary_artifacts(
    round_dir: Path,
    *,
    metric_set: str,
    summary_name: str,
    summary: dict[str, Any],
) -> None:
    summary_base = Path(summary_name)
    (round_dir / f"{summary_base}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (round_dir / f"{summary_base}.md").write_text(
        metrics_to_markdown(f"# {metric_set.upper()} RAGAS Summary", summary),
        encoding="utf-8",
    )
    write_metric_plot(
        pd.DataFrame([summary]),
        round_dir / f"{summary_base}.png",
        title=f"{metric_set.upper()} RAGAS Summary",
    )


def _rerun_ragas_single_row(
    *,
    round_dir: Path,
    rag_results: list[dict[str, Any]],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tmp_root = round_dir / "_single_row_ragas_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    row_timeout = _env_int("RAGAS_ROW_TIMEOUT", 600)
    row_attempts = _env_int("RAGAS_ROW_ATTEMPTS", 4)
    row_backoff = _env_int("RAGAS_ROW_BACKOFF_SECONDS", 30)

    raw_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rag_results, start=1):
        row_dir = tmp_root / f"row_{idx:03d}"
        row_dir.mkdir(parents=True, exist_ok=True)
        answers_path = row_dir / "answers.jsonl"
        scores_path = row_dir / "ragas_scores.csv"
        summary_path = row_dir / "ragas_summary.json"
        if scores_path.exists() and summary_path.exists():
            existing_df = pd.read_csv(scores_path)
            if not existing_df.empty:
                raw_rows.append(existing_df.iloc[0].to_dict())
                print(
                    json.dumps(
                        {
                            "event": "single_row_ragas_resume_existing",
                            "row_index": idx,
                            "question": str(row.get("question", ""))[:120],
                        },
                        ensure_ascii=False,
                    )
                )
                continue
        compare._write_jsonl(answers_path, [row])

        last_exc: Exception | None = None
        for attempt in range(1, row_attempts + 1):
            try:
                row_df, _ = _run_ragas_subprocess_with_timeout(
                    answers_path,
                    row_dir,
                    "rag",
                    model_name,
                    timeout_seconds=row_timeout,
                )
                if row_df.empty:
                    raise RuntimeError(f"single_row_ragas_empty row={idx}")
                raw_rows.append(row_df.iloc[0].to_dict())
                print(
                    json.dumps(
                        {
                            "event": "single_row_ragas_done",
                            "row_index": idx,
                            "attempt": attempt,
                            "question": str(row.get("question", ""))[:120],
                        },
                        ensure_ascii=False,
                    )
                )
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover - retry path
                last_exc = exc
                print(
                    json.dumps(
                        {
                            "event": "single_row_ragas_retry",
                            "row_index": idx,
                            "attempt": attempt,
                            "question": str(row.get("question", ""))[:120],
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
                if attempt < row_attempts:
                    time.sleep(row_backoff)
        if last_exc is not None:
            raise last_exc

    raw_df = pd.DataFrame(raw_rows)
    compare._write_frame_artifacts(raw_df, round_dir / "ragas_scores.csv", round_dir / "ragas_scores.md")

    rag_df = compare._attach_metadata(raw_df, rag_results)
    compare._write_frame_artifacts(
        rag_df,
        round_dir / "ragas_scores_with_meta.csv",
        round_dir / "ragas_scores_with_meta.md",
    )

    rag_summary = compute_metrics(
        raw_df,
        method="RAG",
        extra={
            "metric_set": "rag",
            "model": model_name,
            "answers_path": str(round_dir / "rag_answers.jsonl"),
        },
    )
    rag_summary.update(
        {
            key: value
            for key, value in compare._operational_summary(rag_results, "RAG").items()
            if key not in {"method", "rows"}
        }
    )
    rag_summary["method"] = "RAG"
    rag_summary["eval_mode"] = "ragas"
    _write_ragas_summary_artifacts(
        round_dir,
        metric_set="rag",
        summary_name="ragas_summary",
        summary=rag_summary,
    )
    shutil.rmtree(tmp_root, ignore_errors=True)
    return rag_df, rag_summary


def _write_round_outputs(
    round_dir: Path,
    *,
    rag_summary: dict[str, Any],
    rag_df: pd.DataFrame | None,
    source_round_dir: Path,
    sample_model_slug: str,
    group_name: str,
    group_index: int,
    round_index: int,
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    if rag_df is not None and not rag_df.empty:
        compare._write_frame_artifacts(rag_df, round_dir / "ragas_scores_with_meta.csv", round_dir / "ragas_scores_with_meta.md")

    payload = {
        "method": "RAG",
        "group_name": group_name,
        "group_index": group_index,
        "round": round_index,
        "top_k": FINAL_CONTEXT_TOP_K,
        "sample_source_round_dir": str(source_round_dir),
        "sample_source_model_slug": sample_model_slug,
        "rag": rag_summary,
    }
    _write_json(round_dir / "summary.json", payload)
    summary_md = "\n\n".join(
        [
            metrics_to_markdown(
                "# RAG Stability Round Summary",
                {
                    "group_name": group_name,
                    "group_index": group_index,
                    "round": round_index,
                    "top_k": FINAL_CONTEXT_TOP_K,
                    "sample_source_round_dir": str(source_round_dir),
                    "sample_source_model_slug": sample_model_slug,
                },
            ),
            metrics_to_markdown("## RAG Metrics", rag_summary),
        ]
    )
    (round_dir / "summary.md").write_text(summary_md, encoding="utf-8")


def _aggregate_model_outputs(
    output_root: Path,
    model_spec: dict[str, str],
    round_records: list[dict[str, Any]],
) -> None:
    model_dir = output_root / "models" / model_spec["slug"]
    model_dir.mkdir(parents=True, exist_ok=True)
    per_round_df = pd.DataFrame(round_records)
    compare._write_frame_artifacts(per_round_df, model_dir / "per_round_results.csv", model_dir / "per_round_results.md")
    compare._write_frame_artifacts(per_round_df, model_dir / "sampling_rounds.csv", model_dir / "sampling_rounds.md")
    aggregate_df = compare._aggregate_sampling_rounds(round_records)
    compare._write_frame_artifacts(aggregate_df, model_dir / "model_summary.csv", model_dir / "model_summary.md")
    compare._write_frame_artifacts(aggregate_df, model_dir / "comparison.csv", model_dir / "comparison.md")
    if not aggregate_df.empty:
        write_metric_plot(aggregate_df, model_dir / "comparison_summary.png", title=f"{model_spec['label']} RAG Stability Summary")
    leaderboard_df = compare._build_leaderboard(aggregate_df)
    compare._write_frame_artifacts(leaderboard_df, model_dir / "leaderboard.csv", model_dir / "leaderboard.md")
    summary = {
        "model_label": model_spec["label"],
        "model": model_spec["model"],
        "method": "RAG",
        "rounds": int(len(per_round_df)),
        "top_k": FINAL_CONTEXT_TOP_K,
        "per_round_results_path": str(model_dir / "per_round_results.csv"),
        "model_summary_path": str(model_dir / "model_summary.csv"),
        "leaderboard_path": str(model_dir / "leaderboard.csv"),
    }
    _write_json(model_dir / "summary.json", summary)
    (model_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Selected RAG Stability Model Summary", summary),
                "## Aggregate Summary\n\n" + frame_to_markdown(aggregate_df),
                "## Leaderboard\n\n" + frame_to_markdown(leaderboard_df),
            ]
        ),
        encoding="utf-8",
    )


def _aggregate_root_outputs(
    output_root: Path,
    benchmark_models: list[dict[str, str]],
    all_round_records: list[dict[str, Any]],
) -> None:
    per_round_df = pd.DataFrame(all_round_records)
    compare._write_frame_artifacts(per_round_df, output_root / "per_round_results.csv", output_root / "per_round_results.md")
    compare._write_frame_artifacts(per_round_df, output_root / "sampling_rounds.csv", output_root / "sampling_rounds.md")
    aggregate_df = compare._aggregate_sampling_rounds(all_round_records)
    compare._write_frame_artifacts(aggregate_df, output_root / "model_summary.csv", output_root / "model_summary.md")
    compare._write_frame_artifacts(aggregate_df, output_root / "comparison.csv", output_root / "comparison.md")
    if not aggregate_df.empty:
        write_metric_plot(aggregate_df, output_root / "comparison_summary.png", title="Selected RAG Stability Summary")
    leaderboard_df = compare._build_leaderboard(aggregate_df)
    compare._write_frame_artifacts(leaderboard_df, output_root / "leaderboard.csv", output_root / "leaderboard.md")
    summary = {
        "method": "RAG",
        "top_k": FINAL_CONTEXT_TOP_K,
        "benchmark_models": benchmark_models,
        "round_count": int(len(per_round_df)),
        "per_round_results_path": str(output_root / "per_round_results.csv"),
        "model_summary_path": str(output_root / "model_summary.csv"),
        "leaderboard_path": str(output_root / "leaderboard.csv"),
    }
    _write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Selected RAG Stability Summary", summary),
                "## Aggregate Summary\n\n" + frame_to_markdown(aggregate_df),
                "## Leaderboard\n\n" + frame_to_markdown(leaderboard_df),
            ]
        ),
        encoding="utf-8",
    )


def _backup_round_files(round_dir: Path, model_slug: str, group_name: str, round_name: str) -> None:
    backup_root = round_dir.parents[4] / "_failed_row_repair_backups"
    stamp = _timestamp()
    backup_dir = backup_root / model_slug / group_name / round_name / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "rag_answers.jsonl",
        "ragas_scores.csv",
        "ragas_scores.md",
        "ragas_scores_with_meta.csv",
        "ragas_scores_with_meta.md",
        "ragas_summary.json",
        "ragas_summary.md",
        "ragas_summary.png",
        "summary.json",
        "summary.md",
    ]:
        src = round_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)


def _rerun_failed_rows_for_round(
    *,
    round_dir: Path,
    model_settings: Any,
    model_label: str,
    model_slug: str,
    group_name: str,
    group_index: int,
    sample_source_round_dir: Path,
    sample_source_model_slug: str,
) -> dict[str, Any]:
    answers_path = round_dir / "rag_answers.jsonl"
    if not answers_path.exists():
        raise FileNotFoundError(str(answers_path))

    rag_results = compare._load_jsonl(answers_path)
    failed_indexes = [idx for idx, row in enumerate(rag_results) if str(row.get("status", "ok")) != "ok"]
    existing_summary = _load_existing_rag_summary(round_dir)
    if not failed_indexes and existing_summary and int(existing_summary.get("failed_rows", 0) or 0) == 0:
        return {
            "model_label": model_label,
            "group_name": group_name,
            "round_dir": str(round_dir),
            "failed_before": 0,
            "failed_after": int(existing_summary.get("failed_rows", 0)) if existing_summary else 0,
            "repaired_rows": 0,
        }

    _backup_round_files(round_dir, model_slug, group_name, round_dir.name)

    for idx in failed_indexes:
        source_row = dict(rag_results[idx])
        repaired_row = compare._build_rag_results(
            [source_row],
            model_settings,
            top_k=FINAL_CONTEXT_TOP_K,
        )[0]
        rag_results[idx] = repaired_row

    compare._write_jsonl(answers_path, rag_results)
    ragas_timeout = _env_int("RAGAS_SUBPROCESS_TIMEOUT", 1200)
    try:
        rag_df, rag_summary = _run_ragas_subprocess_with_timeout(
            answers_path,
            round_dir,
            "rag",
            model_settings.llm_model,
            timeout_seconds=ragas_timeout,
        )
        rag_df = compare._attach_metadata(rag_df, rag_results)
        compare._write_frame_artifacts(
            rag_df,
            round_dir / "ragas_scores_with_meta.csv",
            round_dir / "ragas_scores_with_meta.md",
        )
        rag_summary.update(
            {
                key: value
                for key, value in compare._operational_summary(rag_results, "RAG").items()
                if key not in {"method", "rows"}
            }
        )
        rag_summary["method"] = "RAG"
        rag_summary["eval_mode"] = "ragas"
    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                {
                    "event": "ragas_round_timeout_fallback_single_row",
                    "round_dir": str(round_dir),
                    "timeout_seconds": ragas_timeout,
                    "model": model_settings.llm_model,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        rag_df, rag_summary = _rerun_ragas_single_row(
            round_dir=round_dir,
            rag_results=rag_results,
            model_name=model_settings.llm_model,
        )

    _write_round_outputs(
        round_dir,
        rag_summary=rag_summary,
        rag_df=rag_df,
        source_round_dir=sample_source_round_dir,
        sample_model_slug=sample_source_model_slug,
        group_name=group_name,
        group_index=group_index,
        round_index=_round_index(round_dir.name),
    )
    return {
        "model_label": model_label,
        "group_name": group_name,
        "round_dir": str(round_dir),
        "failed_before": len(failed_indexes),
        "failed_after": int(rag_summary.get("failed_rows", 0)),
        "repaired_rows": len(failed_indexes),
    }


def _rebuild_aggregates(output_root: Path, benchmark_models: list[dict[str, str]], source_sampling_root: Path) -> None:
    all_round_records: list[dict[str, Any]] = []
    for spec in benchmark_models:
        model_round_records: list[dict[str, Any]] = []
        model_root = output_root / "sampling"
        for group_dir in sorted(path for path in model_root.glob("group_*") if path.is_dir()):
            group_name = group_dir.name
            group_index = int(group_name.split("_")[-1])
            seed_map = _load_group_round_seed_map(source_sampling_root / group_name)
            model_dir = group_dir / "models" / spec["slug"]
            if not model_dir.exists():
                continue
            for round_dir in sorted(path for path in model_dir.glob("round_*") if path.is_dir()):
                rag_summary = _load_existing_rag_summary(round_dir)
                if rag_summary is None:
                    continue
                sample_rows = _sample_rows_from_direct_answers(round_dir / "direct_answers.jsonl")
                round_index = _round_index(round_dir.name)
                round_seed = seed_map.get(round_index, round_index)
                record = compare._round_record(rag_summary, round_index, round_seed, sample_rows, spec["label"], spec["model"])
                record["group_name"] = group_name
                record["group_index"] = group_index
                record["global_round"] = ((group_index - 1) * 5) + round_index
                model_round_records.append(record)
                all_round_records.append(record)
        _aggregate_model_outputs(output_root, spec, model_round_records)
    _aggregate_root_outputs(output_root, benchmark_models, all_round_records)


def repair_selected_rag_failed_rows(
    *,
    output_root: Path,
    model_selector: str,
    round_targets: list[tuple[str, str]],
    actual_model: str | None = None,
) -> list[dict[str, Any]]:
    load_dotenv()
    experiment_config = _load_json(output_root / "experiment_config.json")
    source_sampling_root = Path(experiment_config["source_sampling_root"])
    benchmark_models: list[dict[str, str]] = experiment_config["benchmark_models"]
    sample_source_model_slug = experiment_config["sample_source_model_slug"]

    target_spec = None
    for spec in benchmark_models:
        if model_selector in {spec["label"], spec["model"], spec["slug"]}:
            target_spec = spec
            break
    if target_spec is None:
        raise ValueError(f"unknown model selector: {model_selector}")

    settings = compare.Settings()
    resolved_actual_model = actual_model or target_spec["model"]
    model_settings = compare._settings_with_model(settings, resolved_actual_model)

    repaired: list[dict[str, Any]] = []
    for group_name, round_name in round_targets:
        round_dir = output_root / "sampling" / group_name / "models" / target_spec["slug"] / round_name
        if not round_dir.exists():
            raise FileNotFoundError(str(round_dir))
        existing_summary_path = round_dir / "summary.json"
        sample_source_round_dir = None
        if existing_summary_path.exists():
            try:
                payload = _load_json(existing_summary_path)
                sample_source_round = payload.get("sample_source_round_dir")
                if sample_source_round:
                    sample_source_round_dir = Path(sample_source_round)
            except Exception:
                sample_source_round_dir = None
        if sample_source_round_dir is None:
            sample_source_round_dir = source_sampling_root / group_name / "models" / sample_source_model_slug / round_name

        repaired.append(
            _rerun_failed_rows_for_round(
                round_dir=round_dir,
                model_settings=model_settings,
                model_label=target_spec["label"],
                model_slug=target_spec["slug"],
                group_name=group_name,
                group_index=int(group_name.split("_")[-1]),
                sample_source_round_dir=sample_source_round_dir,
                sample_source_model_slug=sample_source_model_slug,
            )
        )

    _rebuild_aggregates(output_root, benchmark_models, source_sampling_root)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair failed RAG rows inside selected stability rounds and rebuild aggregates.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True, help="Model label, slug, or API model name.")
    parser.add_argument("--actual-model", default=None, help="Optional API model override while preserving the original display label/slug.")
    parser.add_argument("--round-target", action="append", required=True, type=_parse_target, help="Target round like group_07/round_002")
    args = parser.parse_args()

    repaired = repair_selected_rag_failed_rows(
        output_root=Path(args.output_root),
        model_selector=args.model,
        round_targets=list(args.round_target),
        actual_model=args.actual_model,
    )
    print(json.dumps(repaired, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
