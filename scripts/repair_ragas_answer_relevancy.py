from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.metrics._answer_relevance import ResponseRelevanceInput


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.metrics import build_category_breakdown, compute_metrics, frame_to_markdown, metrics_to_markdown
from metrics.plotting import write_metric_plot
from pipelines.compare import (
    _aggregate_sampling_rounds,
    _attach_metadata,
    _build_comparison_frame,
    _build_leaderboard,
    _write_frame_artifacts,
)
from pipelines.ragas_eval import (
    LocalHashEmbeddings,
    _answer_relevancy_metric,
    _load_jsonl,
    _resolve_base_url,
)
from pipelines.ragas_json_compat import install_ragas_fenced_json_compat
from ragkb.config import Settings


def merge_answer_relevancy_scores(existing_scores: pd.DataFrame, repaired_scores: pd.DataFrame) -> pd.DataFrame:
    merged = existing_scores.copy()
    if "answer_relevancy" not in merged.columns:
        merged["answer_relevancy"] = pd.NA

    repair = repaired_scores.copy()
    repair = repair[["user_input", "answer_relevancy"]].rename(columns={"answer_relevancy": "answer_relevancy_repaired"})

    merged = merged.merge(repair, on="user_input", how="left")
    merged["answer_relevancy"] = merged["answer_relevancy"].where(
        merged["answer_relevancy"].notna(),
        merged["answer_relevancy_repaired"],
    )
    return merged.drop(columns=["answer_relevancy_repaired"])


def replace_round_summary(round_frame: pd.DataFrame, method: str, summary: dict[str, Any]) -> pd.DataFrame:
    updated = round_frame.copy()
    mask = updated["method"].astype(str).str.upper() == method.upper()
    if not mask.any():
        raise ValueError(f"round frame does not contain method={method}")
    row_index = updated.index[mask][0]
    for key, value in summary.items():
        updated.loc[row_index, key] = value
    return updated


async def _score_answer_relevancy_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    strictness: int,
    max_attempts: int,
    repair_log_path: Path,
) -> pd.DataFrame:
    install_ragas_fenced_json_compat()
    load_dotenv(REPO_ROOT / ".env")
    os.environ["LLM_MODEL"] = model_name
    settings = Settings()

    llm_kwargs = {
        "model": model_name,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "request_timeout": 120.0,
        "max_retries": 1,
        "temperature": 0.01,
    }
    base_url = _resolve_base_url()
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)
    metric = _answer_relevancy_metric(strictness=strictness)
    metric.llm = llm
    metric.embeddings = LocalHashEmbeddings(settings)

    existing_repairs: dict[int, dict[str, Any]] = {}
    if repair_log_path.exists():
        prior = pd.read_csv(repair_log_path)
        for record in prior.to_dict(orient="records"):
            row_index = int(record["row_index"])
            existing_repairs[row_index] = record

    repaired_rows: list[dict[str, Any]] = [existing_repairs[index] for index in sorted(existing_repairs)]
    for index, row in enumerate(rows, start=1):
        prior = existing_repairs.get(index)
        if prior and str(prior.get("repair_status", "")) == "ok" and pd.notna(prior.get("answer_relevancy")):
            print(f"[repair_answer_relevancy] row={index}/{len(rows)} status=resume_ok")
            continue

        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        score: float | None = None
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                responses = await metric.question_generation.generate_multiple(
                    data=ResponseRelevanceInput(response=answer),
                    llm=llm,
                    n=strictness,
                )
                score_value = metric._calculate_score(responses, {"user_input": question})
                if pd.notna(score_value):
                    score = round(float(score_value), 4)
                    break
                last_error = "answer_relevancy_nan"
            except Exception as exc:  # pragma: no cover - network/provider failure path
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 4))

        record = {
            "row_index": index,
            "user_input": question,
            "answer_relevancy": score,
            "repair_status": "ok" if score is not None else "error",
            "repair_error": last_error,
        }
        existing_repairs[index] = record
        repaired_rows = [existing_repairs[idx] for idx in sorted(existing_repairs)]
        pd.DataFrame(repaired_rows).to_csv(repair_log_path, index=False)
        print(
            f"[repair_answer_relevancy] row={index}/{len(rows)} status={record['repair_status']}"
            + (f" error={last_error}" if last_error and score is None else "")
        )

    return pd.DataFrame([existing_repairs[idx] for idx in sorted(existing_repairs)])


def _write_round_artifacts(
    round_dir: Path,
    rag_results: list[dict[str, Any]],
    direct_results: list[dict[str, Any]],
    rag_scores: pd.DataFrame,
    rag_metric_summary: dict[str, Any],
    round_summary_payload: dict[str, Any],
) -> None:
    direct_scores = pd.read_csv(round_dir / "direct_ragas_scores.csv")
    direct_metric_summary = json.loads((round_dir / "direct_ragas_summary.json").read_text(encoding="utf-8"))

    (round_dir / "ragas_scores.csv").write_text(rag_scores.to_csv(index=False), encoding="utf-8")
    (round_dir / "ragas_scores.md").write_text(frame_to_markdown(rag_scores), encoding="utf-8")
    (round_dir / "ragas_summary.json").write_text(json.dumps(rag_metric_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (round_dir / "ragas_summary.md").write_text(
        metrics_to_markdown("# RAG RAGAS Summary", rag_metric_summary),
        encoding="utf-8",
    )
    write_metric_plot(pd.DataFrame([rag_metric_summary]), round_dir / "ragas_summary.png", title="RAG RAGAS Summary")

    rag_df_with_meta = _attach_metadata(rag_scores, rag_results)
    direct_df_with_meta = _attach_metadata(direct_scores, direct_results)
    category_breakdown = build_category_breakdown(
        pd.concat([rag_df_with_meta.assign(method="RAG"), direct_df_with_meta.assign(method="Direct")], ignore_index=True)
    )
    category_breakdown.to_csv(round_dir / "category_breakdown.csv", index=False)
    (round_dir / "category_breakdown.md").write_text(frame_to_markdown(category_breakdown), encoding="utf-8")

    comparison_df = _build_comparison_frame("ragas", round_summary_payload["rag"], round_summary_payload["direct"])
    comparison_df.to_csv(round_dir / "comparison.csv", index=False)
    (round_dir / "comparison.md").write_text(frame_to_markdown(comparison_df), encoding="utf-8")
    write_metric_plot(comparison_df, round_dir / "comparison_summary.png", title="RAG vs Direct Comparison")

    (round_dir / "summary.json").write_text(json.dumps(round_summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (round_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown(
                    "# Eval Summary",
                    {
                        "data_path": round_summary_payload["data_path"],
                        "samples": round_summary_payload["samples"],
                        "eval_mode": round_summary_payload["eval_mode"],
                    },
                ),
                "## Comparison\n\n" + frame_to_markdown(comparison_df),
                "## Category Breakdown\n\n" + frame_to_markdown(category_breakdown),
                metrics_to_markdown("## RAG Metrics", round_summary_payload["rag"]),
                metrics_to_markdown("## Direct Metrics", round_summary_payload["direct"]),
            ]
        ),
        encoding="utf-8",
    )


def _rebuild_model_dir(model_dir: Path, rag_summary: dict[str, Any]) -> None:
    per_round = pd.read_csv(model_dir / "per_round_results.csv")
    per_round = replace_round_summary(per_round, "RAG", rag_summary)
    _write_frame_artifacts(per_round, model_dir / "per_round_results.csv", model_dir / "per_round_results.md")
    _write_frame_artifacts(per_round, model_dir / "sampling_rounds.csv", model_dir / "sampling_rounds.md")

    aggregate_df = _aggregate_sampling_rounds(per_round.to_dict(orient="records"))
    _write_frame_artifacts(aggregate_df, model_dir / "model_summary.csv", model_dir / "model_summary.md")
    _write_frame_artifacts(aggregate_df, model_dir / "comparison.csv", model_dir / "comparison.md")
    write_metric_plot(aggregate_df, model_dir / "comparison_summary.png", title="Repeated-Sampling RAG vs Direct Comparison")
    leaderboard_df = _build_leaderboard(aggregate_df)
    _write_frame_artifacts(leaderboard_df, model_dir / "leaderboard.csv", model_dir / "leaderboard.md")

    summary_payload = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    summary_payload["model_summary_path"] = str(model_dir / "model_summary.csv")
    summary_payload["comparison_path"] = str(model_dir / "comparison.csv")
    summary_payload["leaderboard_path"] = str(model_dir / "leaderboard.csv")
    summary_payload["per_round_results_path"] = str(model_dir / "per_round_results.csv")
    summary_payload["rounds_path"] = str(model_dir / "sampling_rounds.csv")
    (model_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "summary.md").write_text(
        "\n\n".join(
            [
                metrics_to_markdown("# Sampling Eval Summary", summary_payload),
                "## Aggregate Comparison\n\n" + frame_to_markdown(aggregate_df),
                "## Leaderboard\n\n" + frame_to_markdown(leaderboard_df),
                "## Round-Level Records\n\n" + frame_to_markdown(per_round),
            ]
        ),
        encoding="utf-8",
    )


def _rebuild_merged_report(report_dir: Path) -> None:
    model_dirs = sorted(path for path in (report_dir / "models").iterdir() if path.is_dir())
    model_summary_frames = [pd.read_csv(model_dir / "model_summary.csv") for model_dir in model_dirs if (model_dir / "model_summary.csv").exists()]
    per_round_frames = [pd.read_csv(model_dir / "per_round_results.csv") for model_dir in model_dirs if (model_dir / "per_round_results.csv").exists()]
    sampling_frames = [pd.read_csv(model_dir / "sampling_rounds.csv") for model_dir in model_dirs if (model_dir / "sampling_rounds.csv").exists()]

    aggregate_df = pd.concat(model_summary_frames, ignore_index=True) if model_summary_frames else pd.DataFrame()
    per_round_df = pd.concat(per_round_frames, ignore_index=True) if per_round_frames else pd.DataFrame()
    sampling_df = pd.concat(sampling_frames, ignore_index=True) if sampling_frames else pd.DataFrame()
    leaderboard_df = _build_leaderboard(aggregate_df)

    _write_frame_artifacts(aggregate_df, report_dir / "model_summary.csv", report_dir / "model_summary.md")
    _write_frame_artifacts(aggregate_df, report_dir / "comparison.csv", report_dir / "comparison.md")
    write_metric_plot(aggregate_df, report_dir / "comparison_summary.png", title="Merged RAG vs Direct Comparison")
    _write_frame_artifacts(leaderboard_df, report_dir / "leaderboard.csv", report_dir / "leaderboard.md")
    _write_frame_artifacts(per_round_df, report_dir / "per_round_results.csv", report_dir / "per_round_results.md")
    _write_frame_artifacts(sampling_df, report_dir / "sampling_rounds.csv", report_dir / "sampling_rounds.md")

    summary_payload = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    summary_payload["model_summary_path"] = str(report_dir / "model_summary.csv")
    summary_payload["leaderboard_path"] = str(report_dir / "leaderboard.csv")
    summary_payload["per_round_results_path"] = str(report_dir / "per_round_results.csv")
    (report_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(
        "# Merged Multi-Model Summary\n\n"
        + frame_to_markdown(leaderboard_df)
        + "\n\n## Model Summary\n\n"
        + frame_to_markdown(aggregate_df),
        encoding="utf-8",
    )


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Repair missing RAG answer_relevancy scores in-place without changing the original judge model or strictness.")
    parser.add_argument("--report-dir", required=True, help="Merged report directory containing models/<slug>/round_001.")
    parser.add_argument("--model-slug", required=True, help="Model slug to repair, e.g. minimax-m2.")
    parser.add_argument("--round-name", default="round_001")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--strictness", type=int, default=3)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    model_dir = report_dir / "models" / args.model_slug
    round_dir = model_dir / args.round_name
    if not round_dir.exists():
        raise FileNotFoundError(str(round_dir))

    round_summary = json.loads((round_dir / "summary.json").read_text(encoding="utf-8"))
    model_name = str(round_summary["rag"]["model"])
    rag_answers = _load_jsonl(round_dir / "rag_answers.jsonl")
    direct_answers = _load_jsonl(round_dir / "direct_answers.jsonl")

    repair_log_path = round_dir / "answer_relevancy_repair_log.csv"
    repaired = await _score_answer_relevancy_rows(
        rag_answers,
        model_name=model_name,
        strictness=args.strictness,
        max_attempts=args.max_attempts,
        repair_log_path=repair_log_path,
    )

    existing_scores = pd.read_csv(round_dir / "ragas_scores.csv")
    merged_scores = merge_answer_relevancy_scores(existing_scores, repaired)

    rag_metric_summary = compute_metrics(
        merged_scores,
        method="RAG",
        extra={
            "metric_set": "rag",
            "model": model_name,
            "answers_path": str(round_dir / "rag_answers.jsonl"),
        },
    )
    round_summary["rag"].update({k: v for k, v in rag_metric_summary.items() if k not in {"method", "rows"}})
    round_summary["rag"]["method"] = "RAG"
    round_summary["rag"]["rows"] = int(round_summary["rag"].get("rows", len(rag_answers)))

    repair_log = repaired[["row_index", "user_input", "answer_relevancy", "repair_status", "repair_error"]]
    repair_log.to_csv(repair_log_path, index=False)
    (round_dir / "answer_relevancy_repair_log.md").write_text(frame_to_markdown(repair_log), encoding="utf-8")

    _write_round_artifacts(
        round_dir=round_dir,
        rag_results=rag_answers,
        direct_results=direct_answers,
        rag_scores=merged_scores,
        rag_metric_summary=rag_metric_summary,
        round_summary_payload=round_summary,
    )
    _rebuild_model_dir(model_dir, round_summary["rag"])
    _rebuild_merged_report(report_dir)
    print(f"Repaired answer_relevancy for {args.model_slug} in {report_dir}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
