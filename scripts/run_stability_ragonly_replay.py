from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.metrics import frame_to_markdown
from metrics.plotting import write_metric_plot
from pipelines import compare
from pipelines.benchmark_protocol import FINAL_CONTEXT_TOP_K, STABILITY_ANCHOR_MODELS
from scripts.aggregate_formal_benchmark import main as aggregate_formal_benchmark  # type: ignore


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_direct_artifacts(source_round_dir: Path, target_round_dir: Path) -> dict[str, Any]:
    target_round_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in (
        "direct_answers.jsonl",
        "direct_ragas_scores.csv",
        "direct_ragas_scores.md",
        "direct_ragas_summary.json",
        "direct_ragas_summary.md",
    ):
        src = source_round_dir / name
        if src.exists():
            shutil.copy2(src, target_round_dir / name)
            copied[name] = str(target_round_dir / name)
    source_summary = _load_json(source_round_dir / "summary.json")
    return {
        "direct_summary": dict(source_summary["direct"]),
        "copied_files": copied,
    }


def _sample_rows_from_direct_answers(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
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


def _load_model_specs(main_dir: Path) -> list[dict[str, str]]:
    config = _load_json(main_dir / "experiment_config.json")
    specs = list(config.get("benchmark_models") or [])
    allowed = set(STABILITY_ANCHOR_MODELS)
    return [spec for spec in specs if spec.get("model") in allowed]


def _load_group_round_seed_map(group_dir: Path) -> dict[tuple[str, int], int]:
    csv_path = group_dir / "per_round_results.csv"
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    if frame.empty or "model" not in frame.columns or "round" not in frame.columns or "seed" not in frame.columns:
        return {}
    seed_map: dict[tuple[str, int], int] = {}
    for row in frame.to_dict(orient="records"):
        try:
            key = (str(row["model"]), int(row["round"]))
            seed_map[key] = int(row["seed"])
        except Exception:
            continue
    return seed_map


def _write_round_outputs(
    round_dir: Path,
    *,
    rag_summary: dict[str, Any],
    direct_summary: dict[str, Any],
    comparison_df: pd.DataFrame,
    reused_direct_round_dir: Path,
) -> None:
    comparison_df.to_csv(round_dir / "comparison.csv", index=False)
    (round_dir / "comparison.md").write_text(frame_to_markdown(comparison_df), encoding="utf-8")
    write_metric_plot(comparison_df, round_dir / "comparison_summary.png", title="RAG Replay vs Reused Direct")
    payload = {
        "top_k": FINAL_CONTEXT_TOP_K,
        "reused_direct_round_dir": str(reused_direct_round_dir),
        "rag": rag_summary,
        "direct": direct_summary,
    }
    _write_json(round_dir / "summary.json", payload)
    (round_dir / "summary.md").write_text("# RAG Replay Round Summary\n\n" + frame_to_markdown(comparison_df), encoding="utf-8")


def _group_summary_from_round_records(out_dir: Path, round_records: list[dict[str, Any]]) -> None:
    rounds_df = pd.DataFrame(round_records)
    rounds_df.to_csv(out_dir / "per_round_results.csv", index=False)
    (out_dir / "per_round_results.md").write_text(frame_to_markdown(rounds_df), encoding="utf-8")
    rounds_df.to_csv(out_dir / "sampling_rounds.csv", index=False)
    (out_dir / "sampling_rounds.md").write_text(frame_to_markdown(rounds_df), encoding="utf-8")

    aggregate_df = compare._aggregate_sampling_rounds(round_records)
    aggregate_df.to_csv(out_dir / "model_summary.csv", index=False)
    (out_dir / "model_summary.md").write_text(frame_to_markdown(aggregate_df), encoding="utf-8")
    aggregate_df.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(frame_to_markdown(aggregate_df), encoding="utf-8")
    write_metric_plot(aggregate_df, out_dir / "comparison_summary.png", title="RAG Replay Group Comparison")
    leaderboard_df = compare._build_leaderboard(aggregate_df)
    leaderboard_df.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "leaderboard.md").write_text(frame_to_markdown(leaderboard_df), encoding="utf-8")

    payload = {
        "top_k": FINAL_CONTEXT_TOP_K,
        "round_count": int(rounds_df["round"].nunique()) if not rounds_df.empty and "round" in rounds_df.columns else 0,
    }
    _write_json(out_dir / "summary.json", payload)
    (out_dir / "summary.md").write_text("# RAG Replay Group Summary\n\n" + frame_to_markdown(leaderboard_df), encoding="utf-8")


def run_stability_ragonly_replay(
    *,
    main_dir: Path,
    source_sampling_root: Path,
    output_root: Path,
    eval_mode: str = "ragas",
    aggregate: bool = True,
) -> dict[str, Any]:
    load_dotenv()
    settings = compare.Settings()
    model_specs = _load_model_specs(main_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sampling_out = output_root / "sampling"
    sampling_out.mkdir(parents=True, exist_ok=True)

    all_groups = sorted(path for path in source_sampling_root.glob("group_*") if path.is_dir())
    for group_dir in all_groups:
        group_out = sampling_out / group_dir.name
        (group_out / "models").mkdir(parents=True, exist_ok=True)
        round_records: list[dict[str, Any]] = []
        seed_map = _load_group_round_seed_map(group_dir)

        for spec in model_specs:
            source_model_dir = group_dir / "models" / spec["slug"]
            if not source_model_dir.exists():
                continue
            target_model_dir = group_out / "models" / spec["slug"]
            target_model_dir.mkdir(parents=True, exist_ok=True)
            model_settings = compare._settings_with_model(settings, spec["model"])

            for source_round_dir in sorted(path for path in source_model_dir.glob("round_*") if path.is_dir()):
                target_round_dir = target_model_dir / source_round_dir.name
                direct_info = _copy_direct_artifacts(source_round_dir, target_round_dir)
                sample_rows = _sample_rows_from_direct_answers(source_round_dir / "direct_answers.jsonl")
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
                    rag_df, rag_summary = compare._run_ragas_subprocess(
                        rag_answers_path,
                        target_round_dir,
                        "rag",
                        model_settings.llm_model,
                    )
                    rag_df = compare._attach_metadata(rag_df, rag_results)
                    rag_summary.update({k: v for k, v in compare._operational_summary(rag_results, "RAG").items() if k not in {"method", "rows"}})
                    rag_summary["method"] = "RAG"
                    rag_summary["eval_mode"] = resolved_eval_mode
                else:
                    rag_df = compare.evaluate_local_results(rag_results)
                    rag_summary = compare.compute_metrics(rag_df, method="RAG", extra={"eval_mode": resolved_eval_mode})

                direct_summary = dict(direct_info["direct_summary"])
                direct_summary["reused_from"] = str(source_round_dir)
                comparison_df = compare._build_comparison_frame(resolved_eval_mode, rag_summary, direct_summary)
                _write_round_outputs(
                    target_round_dir,
                    rag_summary=rag_summary,
                    direct_summary=direct_summary,
                    comparison_df=comparison_df,
                    reused_direct_round_dir=source_round_dir,
                )

                round_index = int(source_round_dir.name.split("_")[-1])
                seed = seed_map.get((spec["model"], round_index), round_index)
                round_records.append(compare._round_record(rag_summary, round_index, seed, sample_rows, spec["label"], spec["model"]))
                round_records.append(compare._round_record(direct_summary, round_index, seed, sample_rows, spec["label"], spec["model"]))

        _group_summary_from_round_records(group_out, round_records)

    summary = {
        "main_dir": str(main_dir),
        "source_sampling_root": str(source_sampling_root),
        "output_root": str(output_root),
        "group_count": len(all_groups),
        "model_count": len(model_specs),
        "models": model_specs,
    }
    _write_json(output_root / "summary.json", summary)

    if aggregate:
        final_out = output_root / "final"
        compare_cmd_backup = sys.argv[:]
        try:
            sys.argv = [
                "aggregate_formal_benchmark.py",
                "--main-dir",
                str(main_dir),
                "--sampling-root",
                str(sampling_out),
                "--output-dir",
                str(final_out),
            ]
            aggregate_formal_benchmark()
        finally:
            sys.argv = compare_cmd_backup

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay stability benchmark RAG-only using existing sampling groups and Direct outputs.")
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--source-sampling-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-mode", choices=["auto", "ragas", "local"], default="ragas")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()

    run_stability_ragonly_replay(
        main_dir=Path(args.main_dir),
        source_sampling_root=Path(args.source_sampling_root),
        output_root=Path(args.output_root),
        eval_mode=args.eval_mode,
        aggregate=not args.no_aggregate,
    )


if __name__ == "__main__":
    main()
