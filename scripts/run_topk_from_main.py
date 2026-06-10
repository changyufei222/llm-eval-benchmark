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
from pipelines.benchmark_protocol import FINAL_CONTEXT_TOP_K


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    summary = _load_json(source_round_dir / "summary.json")
    return {
        "direct_summary": dict(summary["direct"]),
        "copied_files": copied,
    }


def _write_round_outputs(
    round_dir: Path,
    *,
    rag_summary: dict[str, Any],
    direct_summary: dict[str, Any],
    comparison_df: pd.DataFrame,
    candidate_top_k: int,
    reused_direct_round_dir: Path,
) -> None:
    comparison_df.to_csv(round_dir / "comparison.csv", index=False)
    (round_dir / "comparison.md").write_text(frame_to_markdown(comparison_df), encoding="utf-8")
    write_metric_plot(comparison_df, round_dir / "comparison_summary.png", title="topK RAG vs Reused Direct")
    summary_payload = {
        "eval_mode": rag_summary.get("eval_mode"),
        "top_k": FINAL_CONTEXT_TOP_K,
        "candidate_top_k": candidate_top_k,
        "reused_direct_round_dir": str(reused_direct_round_dir),
        "rag": rag_summary,
        "direct": direct_summary,
    }
    (round_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (round_dir / "summary.md").write_text(
        "# topK Round Summary\n\n"
        + frame_to_markdown(comparison_df),
        encoding="utf-8",
    )


def run_topk_from_main(
    *,
    main_dir: Path,
    data_path: Path,
    output_dir: Path,
    candidate_top_k: int,
    eval_mode: str = "ragas",
) -> dict[str, Any]:
    load_dotenv()
    settings = compare.Settings()
    rows = compare._load_jsonl(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = _load_json(main_dir / "experiment_config.json")
    benchmark_models = list(experiment_config.get("benchmark_models") or [])

    round_records: list[dict[str, Any]] = []
    model_summary_frames: list[pd.DataFrame] = []

    for spec in benchmark_models:
        model_dir = output_dir / "models" / spec["slug"]
        round_dir = model_dir / "round_001"
        base_round_dir = main_dir / "models" / spec["slug"] / "round_001"
        reused = _copy_direct_artifacts(base_round_dir, round_dir)

        model_settings = compare._settings_with_candidate_top_k(
            compare._settings_with_model(settings, spec["model"]),
            candidate_top_k,
        )

        rag_answers_path = round_dir / "rag_answers.jsonl"
        if rag_answers_path.exists():
            rag_answers_path.unlink()

        rag_results = compare._build_rag_results(rows, model_settings, top_k=FINAL_CONTEXT_TOP_K, persist_path=rag_answers_path)
        resolved_eval_mode, _ = compare._resolve_eval_mode(eval_mode)

        if resolved_eval_mode == "ragas":
            rag_df, rag_summary = compare._run_ragas_subprocess(rag_answers_path, round_dir, "rag", model_settings.llm_model)
            rag_df = compare._attach_metadata(rag_df, rag_results)
            rag_summary.update({k: v for k, v in compare._operational_summary(rag_results, "RAG").items() if k not in {"method", "rows"}})
            rag_summary["method"] = "RAG"
            rag_summary["eval_mode"] = resolved_eval_mode
        else:
            rag_df = compare.evaluate_local_results(rag_results)
            rag_summary = compare.compute_metrics(rag_df, method="RAG", extra={"eval_mode": resolved_eval_mode})

        direct_summary = dict(reused["direct_summary"])
        direct_summary["reused_from"] = str(base_round_dir)
        comparison_df = compare._build_comparison_frame(resolved_eval_mode, rag_summary, direct_summary)
        _write_round_outputs(
            round_dir,
            rag_summary=rag_summary,
            direct_summary=direct_summary,
            comparison_df=comparison_df,
            candidate_top_k=candidate_top_k,
            reused_direct_round_dir=base_round_dir,
        )

        round_records.append(
            compare._round_record(
                rag_summary,
                1,
                int(experiment_config.get("seed", 42)),
                rows,
                model_label=spec["label"],
                model_name=spec["model"],
            )
        )
        round_records.append(
            compare._round_record(
                direct_summary,
                1,
                int(experiment_config.get("seed", 42)),
                rows,
                model_label=spec["label"],
                model_name=spec["model"],
            )
        )
        model_summary_frames.append(comparison_df.assign(model_label=spec["label"], model=spec["model"]))

    per_round_df = pd.DataFrame(round_records)
    model_summary_df = pd.concat(model_summary_frames, ignore_index=True) if model_summary_frames else pd.DataFrame()
    leaderboard_df = compare._build_leaderboard(model_summary_df)

    experiment_payload = {
        "main_dir": str(main_dir),
        "data_path": str(data_path),
        "top_k": FINAL_CONTEXT_TOP_K,
        "candidate_top_k": candidate_top_k,
        "benchmark_models": benchmark_models,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(experiment_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    per_round_df.to_csv(output_dir / "per_round_results.csv", index=False)
    (output_dir / "per_round_results.md").write_text(frame_to_markdown(per_round_df), encoding="utf-8")
    per_round_df.to_csv(output_dir / "sampling_rounds.csv", index=False)
    (output_dir / "sampling_rounds.md").write_text(frame_to_markdown(per_round_df), encoding="utf-8")
    model_summary_df.to_csv(output_dir / "model_summary.csv", index=False)
    (output_dir / "model_summary.md").write_text(frame_to_markdown(model_summary_df), encoding="utf-8")
    model_summary_df.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.md").write_text(frame_to_markdown(model_summary_df), encoding="utf-8")
    if not model_summary_df.empty:
        write_metric_plot(model_summary_df, output_dir / "comparison_summary.png", title="topK Candidate Pool Comparison")
    leaderboard_df.to_csv(output_dir / "leaderboard.csv", index=False)
    (output_dir / "leaderboard.md").write_text(frame_to_markdown(leaderboard_df), encoding="utf-8")

    summary = {
        "main_dir": str(main_dir),
        "data_path": str(data_path),
        "candidate_top_k": candidate_top_k,
        "top_k": FINAL_CONTEXT_TOP_K,
        "model_count": len(benchmark_models),
        "experiment_config_path": str(output_dir / "experiment_config.json"),
        "per_round_results_path": str(output_dir / "per_round_results.csv"),
        "model_summary_path": str(output_dir / "model_summary.csv"),
        "leaderboard_path": str(output_dir / "leaderboard.csv"),
        "models": benchmark_models,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# topK Candidate Pool Summary\n\n"
        + frame_to_markdown(leaderboard_df)
        + "\n\n## Model Summary\n\n"
        + frame_to_markdown(model_summary_df),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a topK ablation by reusing Direct outputs from an existing main benchmark directory.")
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--data-path", default="data/fbtp_eval_fixed_120.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-top-k", type=int, required=True)
    parser.add_argument("--eval-mode", choices=["auto", "ragas", "local"], default="ragas")
    args = parser.parse_args()

    main_dir = Path(args.main_dir)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    run_topk_from_main(
        main_dir=main_dir,
        data_path=data_path,
        output_dir=Path(args.output_dir),
        candidate_top_k=args.candidate_top_k,
        eval_mode=args.eval_mode,
    )


if __name__ == "__main__":
    main()
