from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.metrics import frame_to_markdown, metrics_to_markdown
from metrics.plotting import write_metric_plot
from pipelines import compare
from pipelines.benchmark_protocol import FINAL_CONTEXT_TOP_K


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_rows_from_direct_answers(path: Path) -> list[dict[str, Any]]:
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


def run_stability_selected_rag(
    *,
    source_sampling_root: Path,
    output_root: Path,
    benchmark_models: list[str],
    sample_source_model: str,
    group_start: int = 1,
    group_end: int = 20,
    eval_mode: str = "ragas",
    skip_existing: bool = True,
) -> dict[str, Any]:
    load_dotenv()
    settings = compare.Settings()
    parsed_specs = compare._parse_benchmark_model_specs(benchmark_models, settings.llm_model)
    sample_model_slug = compare._slugify_label(sample_source_model)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_payload = {
        "source_sampling_root": str(source_sampling_root),
        "output_root": str(output_root),
        "benchmark_models": parsed_specs,
        "sample_source_model": sample_source_model,
        "sample_source_model_slug": sample_model_slug,
        "group_start": int(group_start),
        "group_end": int(group_end),
        "top_k": FINAL_CONTEXT_TOP_K,
        "eval_mode": eval_mode,
        "skip_existing": bool(skip_existing),
    }
    _write_json(output_root / "experiment_config.json", experiment_payload)

    all_round_records: list[dict[str, Any]] = []

    for spec in parsed_specs:
        model_round_records: list[dict[str, Any]] = []
        model_settings = compare._settings_with_model(settings, spec["model"])

        for group_index in range(int(group_start), int(group_end) + 1):
            group_name = f"group_{group_index:02d}"
            source_group_dir = source_sampling_root / group_name
            source_model_dir = source_group_dir / "models" / sample_model_slug
            if not source_model_dir.exists():
                raise FileNotFoundError(f"missing sample source model dir: {source_model_dir}")
            seed_map = _load_group_round_seed_map(source_group_dir)

            for source_round_dir in sorted(path for path in source_model_dir.glob("round_*") if path.is_dir()):
                round_index = int(source_round_dir.name.split("_")[-1])
                target_round_dir = output_root / "sampling" / group_name / "models" / spec["slug"] / source_round_dir.name
                sample_rows = _sample_rows_from_direct_answers(source_round_dir / "direct_answers.jsonl")
                round_seed = seed_map.get(round_index, round_index)

                if skip_existing:
                    existing_summary = _load_existing_rag_summary(target_round_dir)
                    if existing_summary is not None:
                        record = compare._round_record(existing_summary, round_index, round_seed, sample_rows, spec["label"], spec["model"])
                        record["group_name"] = group_name
                        record["group_index"] = group_index
                        record["global_round"] = ((group_index - 1) * 5) + round_index
                        model_round_records.append(record)
                        all_round_records.append(record)
                        continue

                rag_answers_path = target_round_dir / "rag_answers.jsonl"
                if rag_answers_path.exists():
                    rag_answers_path.unlink()

                rag_results = compare._build_rag_results(
                    sample_rows,
                    model_settings,
                    top_k=FINAL_CONTEXT_TOP_K,
                    persist_path=rag_answers_path,
                )
                resolved_eval_mode, _ = compare._resolve_eval_mode(eval_mode)

                if resolved_eval_mode == "ragas":
                    rag_df, rag_summary = compare._run_ragas_subprocess(rag_answers_path, target_round_dir, "rag", model_settings.llm_model)
                    rag_df = compare._attach_metadata(rag_df, rag_results)
                    rag_summary.update(
                        {
                            key: value
                            for key, value in compare._operational_summary(rag_results, "RAG").items()
                            if key not in {"method", "rows"}
                        }
                    )
                    rag_summary["method"] = "RAG"
                    rag_summary["eval_mode"] = resolved_eval_mode
                else:
                    rag_df = compare.evaluate_local_results(rag_results)
                    rag_summary = compare.compute_metrics(rag_df, method="RAG", extra={"eval_mode": resolved_eval_mode})

                _write_round_outputs(
                    target_round_dir,
                    rag_summary=rag_summary,
                    rag_df=rag_df,
                    source_round_dir=source_round_dir,
                    sample_model_slug=sample_model_slug,
                    group_name=group_name,
                    group_index=group_index,
                    round_index=round_index,
                )

                record = compare._round_record(rag_summary, round_index, round_seed, sample_rows, spec["label"], spec["model"])
                record["group_name"] = group_name
                record["group_index"] = group_index
                record["global_round"] = ((group_index - 1) * 5) + round_index
                model_round_records.append(record)
                all_round_records.append(record)

        _aggregate_model_outputs(output_root, spec, model_round_records)

    _aggregate_root_outputs(output_root, parsed_specs, all_round_records)
    return {
        "benchmark_models": parsed_specs,
        "output_root": str(output_root),
        "round_count": len(all_round_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected-model RAG-only stability replay from existing sampling groups.")
    parser.add_argument("--source-sampling-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchmark-model", action="append", required=True)
    parser.add_argument("--sample-source-model", default="DeepSeek-V3.2")
    parser.add_argument("--group-start", type=int, default=1)
    parser.add_argument("--group-end", type=int, default=20)
    parser.add_argument("--eval-mode", choices=["auto", "ragas", "local"], default="ragas")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    run_stability_selected_rag(
        source_sampling_root=Path(args.source_sampling_root),
        output_root=Path(args.output_root),
        benchmark_models=list(args.benchmark_model),
        sample_source_model=args.sample_source_model,
        group_start=args.group_start,
        group_end=args.group_end,
        eval_mode=args.eval_mode,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
