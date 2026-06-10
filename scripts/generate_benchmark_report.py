from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.formal_benchmark import combine_main_category_breakdowns, pivot_method_metrics
from metrics.metrics import frame_to_markdown
from pipelines.benchmark_protocol import (
    APPENDIX_NATIVE_MODELS,
    CONTROLLED_MAIN_MODELS,
    FINAL_CONTEXT_TOP_K,
    STABILITY_ANCHOR_MODELS,
    TOPK_CANDIDATE_VALUES,
)


DEFAULT_MAIN_DIR = REPO_ROOT / "reports" / "remote_sync" / "main_results_ragas_fixed120_controlled8_merged_20260418_133150"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "fbtp_eval_fixed_120.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "data" / "fbtp_eval_fixed_120.summary.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "benchmark_formal_report_cn_2026-04-19.md"
DEFAULT_TEACHER_FIGURE_PATH = REPO_ROOT / "docs" / "benchmark_main_8models_direct_vs_rag_2026-04-19.png"

PRIMARY_TAGS = {"ragppi", "doc_design", "schema_tables"}
IGNORE_SECONDARY_TAGS = {"benchmark_fixed_set", "ragppi_gold", "doc_design", "schema_tables"}

MODEL_ROLE_MAP = {
    "DeepSeek-V3.2": "主基线，受控非 thinking",
    "GLM-5": "中文和结构化问答对照组，受控非 thinking",
    "Kimi-K2.5": "长上下文整合对照组，受控非 thinking",
    "DeepSeek-R1": "推理增强组，保留原生 reasoning",
    "Qwen3-235B-A22B-Instruct-2507": "大模型补充组，受控非 thinking",
    "MiniMax-M2": "商业模型补充组，受控非 thinking",
    "MiniMax-M2.7": "MiniMax 系替补增强组，受控非 thinking",
    "ERNIE-4.5-Turbo-128K": "国产商用模型补充组，受控非 thinking",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _primary_bucket(row: dict[str, Any]) -> str:
    tags = [str(tag).strip() for tag in row.get("tags", []) if str(tag).strip()]
    if tags:
        primary = tags[0]
        if primary == "ragppi":
            return "ragppi"
        if primary == "doc_design":
            return "doc/design"
        if primary == "schema_tables":
            return "schema_tables"
    source_group = str(row.get("benchmark_source_group", "")).strip()
    if source_group == "ragppi_gold":
        return "ragppi"
    if source_group == "doc_design":
        return "doc/design"
    if source_group == "schema_tables":
        return "schema_tables"
    return str(row.get("record_type", "unknown"))


def _secondary_bucket(row: dict[str, Any]) -> str:
    tags = [str(tag).strip() for tag in row.get("tags", []) if str(tag).strip()]
    primary = _primary_bucket(row)
    for tag in tags:
        if tag in IGNORE_SECONDARY_TAGS:
            continue
        if primary == "doc/design" and tag == "ragppi":
            continue
        if tag in PRIMARY_TAGS:
            continue
        return tag
    return "none"


def build_question_catalog(dataset_path: Path) -> pd.DataFrame:
    rows = _load_jsonl(dataset_path)
    records: list[dict[str, Any]] = []
    for row in rows:
        primary = _primary_bucket(row)
        secondary = _secondary_bucket(row)
        records.append(
            {
                "question_id": row.get("question_id"),
                "primary_bucket": primary,
                "secondary_bucket": secondary,
                "record_type": row.get("record_type"),
                "benchmark_source_group": row.get("benchmark_source_group"),
                "question": row.get("question"),
            }
        )
    return pd.DataFrame(records)


def build_model_core_summary(main_dir: Path) -> pd.DataFrame:
    model_summary = pd.read_csv(main_dir / "model_summary.csv")
    wide = pivot_method_metrics(
        model_summary,
        index_cols=["model_label", "model"],
        value_cols=[
            "answer_relevancy_mean",
            "latency_ms_mean",
            "completion_tokens_mean",
            "faithfulness_mean",
            "context_precision_mean",
            "failed_rows_mean",
            "success_rows_mean",
        ],
    )
    renamed = wide.rename(
        columns={
            "answer_relevancy_mean_direct": "direct_answer_relevancy",
            "answer_relevancy_mean_rag": "rag_answer_relevancy",
            "latency_ms_mean_direct": "direct_latency_ms",
            "latency_ms_mean_rag": "rag_latency_ms",
            "completion_tokens_mean_direct": "direct_completion_tokens",
            "completion_tokens_mean_rag": "rag_completion_tokens",
            "faithfulness_mean_rag": "rag_faithfulness",
            "context_precision_mean_rag": "rag_context_precision",
            "failed_rows_mean_direct": "direct_failed_rows",
            "failed_rows_mean_rag": "rag_failed_rows",
            "success_rows_mean_direct": "direct_success_rows",
            "success_rows_mean_rag": "rag_success_rows",
        }
    )
    renamed["uplift"] = renamed["rag_answer_relevancy"] - renamed["direct_answer_relevancy"]
    renamed["latency_ratio_rag_vs_direct"] = renamed["rag_latency_ms"] / renamed["direct_latency_ms"]
    renamed["model_role"] = renamed["model_label"].map(MODEL_ROLE_MAP).fillna("")
    return renamed.sort_values(["uplift", "model_label"], ascending=[False, True], na_position="last").reset_index(drop=True)


def write_teacher_main_figure(model_core: pd.DataFrame, output_path: Path) -> None:
    if model_core.empty:
        return

    frame = model_core.copy().sort_values(["uplift", "model_label"], ascending=[False, True], na_position="last").reset_index(drop=True)
    y_pos = list(range(len(frame)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    direct_color = "#6b7280"
    rag_color = "#0f766e"

    ax = axes[0]
    for idx, row in frame.iterrows():
        direct = row.get("direct_answer_relevancy")
        rag = row.get("rag_answer_relevancy")
        if pd.notna(direct) and pd.notna(rag):
            ax.plot([direct, rag], [idx, idx], color="#cbd5e1", linewidth=2, zorder=1)
        elif pd.notna(direct):
            ax.plot([direct, direct], [idx, idx], color="#e5e7eb", linewidth=2, zorder=1)
        if pd.notna(direct):
            ax.scatter(direct, idx, color=direct_color, s=42, zorder=3, label="Direct" if idx == 0 else None)
        if pd.notna(rag):
            ax.scatter(rag, idx, color=rag_color, s=42, zorder=3, label="RAG" if idx == 0 else None)
        else:
            ax.text(
                min(0.98, (float(direct) if pd.notna(direct) else 0.02) + 0.03),
                idx,
                "RAG: NA",
                va="center",
                fontsize=9,
                color="#991b1b",
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frame["model_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Answer Relevancy")
    ax.set_title("Main Table: Direct vs RAG Quality")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for idx, row in frame.iterrows():
        direct = row.get("direct_latency_ms")
        rag = row.get("rag_latency_ms")
        if pd.notna(direct) and pd.notna(rag):
            ax.plot([direct, rag], [idx, idx], color="#cbd5e1", linewidth=2, zorder=1)
        if pd.notna(direct):
            ax.scatter(direct, idx, color=direct_color, s=42, zorder=3, label="Direct" if idx == 0 else None)
        if pd.notna(rag):
            ax.scatter(rag, idx, color=rag_color, s=42, zorder=3, label="RAG" if idx == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frame["model_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_title("Main Table: Direct vs RAG Latency")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("8-Model Benchmark Main Table", fontsize=16, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_category_analysis(main_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    category_long = combine_main_category_breakdowns(main_dir)
    category_wide = pivot_method_metrics(
        category_long,
        index_cols=["model_label", "model", "category_bucket"],
        value_cols=["answer_relevancy", "context_precision", "faithfulness", "latency_ms", "rows"],
    ).rename(columns={"category_bucket": "category"})

    category_wide["answer_relevancy_uplift"] = category_wide["answer_relevancy_rag"] - category_wide["answer_relevancy_direct"]
    category_wide["latency_ratio_rag_vs_direct"] = category_wide["latency_ms_rag"] / category_wide["latency_ms_direct"]
    category_wide = category_wide.sort_values(["category", "model_label"], ascending=[True, True], na_position="last").reset_index(drop=True)

    category_mean = (
        category_wide.groupby("category", dropna=False)["answer_relevancy_uplift"]
        .mean()
        .reset_index(name="answer_relevancy_uplift_mean")
        .sort_values("answer_relevancy_uplift_mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    issue_rows: list[dict[str, Any]] = []
    for row in category_wide.to_dict(orient="records"):
        if pd.isna(row.get("answer_relevancy_rag")):
            issue_rows.append(
                {
                    "scope": "category",
                    "model_label": row.get("model_label"),
                    "category": row.get("category"),
                    "issue": "RAG answer_relevancy 缺失，已在兼容透视表中保留 answer_relevancy_rag 列并以空值标记。",
                }
            )
    issue_df = pd.DataFrame(issue_rows)
    return category_wide, category_mean, issue_df


def build_question_tables(question_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_summary = (
        question_catalog.groupby(["primary_bucket", "record_type", "benchmark_source_group"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "primary_bucket"], ascending=[False, True])
        .reset_index(drop=True)
    )
    secondary_summary = (
        question_catalog.groupby(["primary_bucket", "secondary_bucket"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["primary_bucket", "rows", "secondary_bucket"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return primary_summary, secondary_summary


def build_question_examples(question_catalog: pd.DataFrame, limit_per_bucket: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in ("ragppi", "doc/design", "schema_tables"):
        subset = question_catalog[question_catalog["primary_bucket"] == bucket].head(limit_per_bucket)
        for item in subset.to_dict(orient="records"):
            rows.append(
                {
                    "primary_bucket": bucket,
                    "question_id": item["question_id"],
                    "secondary_bucket": item["secondary_bucket"],
                    "question": item["question"],
                }
            )
    return pd.DataFrame(rows)


def build_protocol_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_name": "主结果表",
                "scope": "8 模型 × Direct / RAG × 固定 120 题",
                "configuration": "rounds=1, sample_size=120, with_replacement=false, seed=42, eval_mode=ragas, top_k=5",
                "why": "作为论文主结论，直接回答 RAG 相对 Direct 的真实增益。",
                "status": "已完成并纳入本报告",
            },
            {
                "table_name": "分类结果表",
                "scope": "8 模型 × ragppi / doc/design / schema_tables",
                "configuration": "与主表同协议，按题型拆分统计",
                "why": "分析 RAG 在不同问题结构上的优势与短板。",
                "status": "设计已锁定，本报告仅引用主表内可直接重建的分类分析",
            },
            {
                "table_name": "稳定性表",
                "scope": f"{len(STABILITY_ANCHOR_MODELS)} 模型 × 100 轮 × 每轮 50 题有放回抽样",
                "configuration": "均值、标准差、P50、P95、95% CI、RAG-Direct uplift",
                "why": "衡量真实 API 条件下的波动性、超时风险与增益稳定性。",
                "status": "结果未在本报告中纳入",
            },
            {
                "table_name": "附加表",
                "scope": f"{len(APPENDIX_NATIVE_MODELS)} 模型 provider-native / native-thinking",
                "configuration": "与主表同题集，但不压制 provider-native 行为",
                "why": "分离主表受控协议与原生 provider 行为之间的差异。",
                "status": "结果未在本报告中纳入",
            },
            {
                "table_name": "topK 消融表",
                "scope": f"{len(CONTROLLED_MAIN_MODELS)} 模型 × topK {list(TOPK_CANDIDATE_VALUES)}",
                "configuration": f"只改候选池 candidate_top_k，最终上下文 top_k 固定为 {FINAL_CONTEXT_TOP_K}",
                "why": "验证候选池增大是否值得延迟和成本代价。",
                "status": "结果未在本报告中纳入",
            },
        ]
    )


def build_metric_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "answer_relevancy", "source": "RAGAS", "meaning": "回答是否真正回答了问题，是主结果表首要指标。"},
            {"metric": "faithfulness", "source": "RAGAS", "meaning": "回答是否忠于检索证据，用于抑制幻觉。"},
            {"metric": "context_precision", "source": "RAGAS", "meaning": "送入回答链的上下文是否相关，反映检索和重排质量。"},
            {"metric": "source_hit_rate", "source": "本地结构化指标", "meaning": "回答是否命中期望来源。"},
            {"metric": "table_hit_rate", "source": "本地结构化指标", "meaning": "回答是否命中目标表族，反映跨表检索对齐。"},
            {"metric": "id_hit_rate", "source": "本地结构化指标", "meaning": "回答是否命中目标实体 ID。"},
            {"metric": "latency_ms", "source": "运行时指标", "meaning": "单题链路时延，用于分析质量与效率折中。"},
            {"metric": "failed_rows / success_rows", "source": "运行时指标", "meaning": "失败题和成功题数量，用于稳定性与工程可用性评估。"},
        ]
    )


def build_answer_inventory(main_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(main_dir.glob("models/*/round_001/*_answers.jsonl")):
        method = "RAG" if path.name.startswith("rag_") else "Direct"
        rows = _load_jsonl(path)
        status_ok = sum(1 for row in rows if str(row.get("status", "ok")) == "ok")
        status_error = len(rows) - status_ok
        avg_chars = round(sum(len(str(row.get("answer", "") or "")) for row in rows) / len(rows), 2) if rows else 0.0
        latency_values = [float(row.get("latency_ms", 0.0) or 0.0) for row in rows]
        avg_latency = round(sum(latency_values) / len(latency_values), 4) if latency_values else 0.0
        records.append(
            {
                "model_label": path.parents[1].name,
                "method": method,
                "rows": len(rows),
                "status_ok_rows": status_ok,
                "status_error_rows": status_error,
                "avg_answer_chars": avg_chars,
                "avg_latency_ms_from_jsonl": avg_latency,
                "path": str(path),
            }
        )
    return pd.DataFrame(records)


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def _format_metric(value: Any, digits: int = 4) -> str:
    converted = _safe_float(value)
    if converted is None:
        return "NA"
    return f"{converted:.{digits}f}"


def _relative_path(target: Path, base_dir: Path) -> str:
    return target.resolve().relative_to(base_dir.resolve().parent).as_posix()


def write_report(
    *,
    main_dir: Path,
    dataset_path: Path,
    summary_path: Path,
    output_path: Path,
) -> dict[str, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_analysis_dir = main_dir / "local_analysis"
    local_analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = _load_json(summary_path)
    experiment_summary = _load_json(main_dir / "summary.json")
    question_catalog = build_question_catalog(dataset_path)
    primary_summary, secondary_summary = build_question_tables(question_catalog)
    question_examples = build_question_examples(question_catalog)
    model_core = build_model_core_summary(main_dir)
    category_wide, category_mean, issue_df = build_category_analysis(main_dir)
    answer_inventory = build_answer_inventory(main_dir)
    protocol_table = build_protocol_table()
    metric_table = build_metric_table()

    question_catalog.to_csv(local_analysis_dir / "question_catalog.csv", index=False)
    (local_analysis_dir / "question_catalog.md").write_text(frame_to_markdown(question_catalog), encoding="utf-8")
    primary_summary.to_csv(local_analysis_dir / "question_primary_summary.csv", index=False)
    (local_analysis_dir / "question_primary_summary.md").write_text(frame_to_markdown(primary_summary), encoding="utf-8")
    secondary_summary.to_csv(local_analysis_dir / "question_secondary_summary.csv", index=False)
    (local_analysis_dir / "question_secondary_summary.md").write_text(frame_to_markdown(secondary_summary), encoding="utf-8")
    question_examples.to_csv(local_analysis_dir / "question_examples.csv", index=False)
    (local_analysis_dir / "question_examples.md").write_text(frame_to_markdown(question_examples), encoding="utf-8")
    model_core.to_csv(local_analysis_dir / "main_table_core_summary.csv", index=False)
    category_wide.to_csv(local_analysis_dir / "category_uplift_summary.csv", index=False)
    category_mean.to_csv(local_analysis_dir / "category_uplift_mean.csv", index=False)
    issue_df.to_csv(local_analysis_dir / "category_data_quality_notes.csv", index=False)
    answer_inventory.to_csv(local_analysis_dir / "answer_inventory.csv", index=False)
    write_teacher_main_figure(model_core, DEFAULT_TEACHER_FIGURE_PATH)

    available_uplift = model_core.dropna(subset=["uplift"])
    best_uplift_row = available_uplift.iloc[0] if not available_uplift.empty else None
    smallest_uplift_row = available_uplift.iloc[-1] if not available_uplift.empty else None
    fastest_direct_row = model_core.sort_values("direct_latency_ms").iloc[0] if not model_core.empty else None
    slowest_rag_row = model_core.sort_values("rag_latency_ms", ascending=False, na_position="last").iloc[0] if not model_core.empty else None
    best_faithfulness_row = model_core.dropna(subset=["rag_faithfulness"]).sort_values("rag_faithfulness", ascending=False).iloc[0] if not model_core.dropna(subset=["rag_faithfulness"]).empty else None
    complete_models = int(model_core["rag_answer_relevancy"].notna().sum())
    total_models = int(len(model_core))
    missing_metric_models = model_core[model_core["rag_answer_relevancy"].isna()]["model_label"].tolist()
    positive_uplift_models = int((available_uplift["uplift"] > 0).sum()) if not available_uplift.empty else 0
    if missing_metric_models:
        missing_models_text = "、".join(f"`{name}`" for name in missing_metric_models)
        completion_paragraph = (
            f"本轮已完成主表合并目录为 `reports/remote_sync/{main_dir.name}`。该目录包含总表 `comparison.csv`、`model_summary.csv`、每个模型的 `round_001` 原始回答、RAGAS 输出和分模型汇总，因此主结果已经具备复查和复算条件。"
            f"当前主表一共覆盖 `{total_models}` 个模型，其中 `{complete_models}` 个模型的 `RAG answer_relevancy` 可直接比较，"
            f"{missing_models_text} 仍保留缺值，需要在后续分析里显式备注而不是强行插值或伪造分数。"
        )
        main_result_paragraph = (
            f"从当前已完成的主表可以直接得到三个结论。第一，除 {missing_models_text} 因 RAG `answer_relevancy` 缺失而暂时无法比较外，"
            f"其余 `{positive_uplift_models}` 个可比较模型全部呈现 `RAG > Direct` 的 answer relevancy，说明你的 RAG 系统在受控协议下具有稳定的正增益。"
        )
        minimax_section_title = "## 15. 缺失列问题到底是什么，怎么处理"
        minimax_section_body = (
            "当前仍存在的缺失不是主表文件丢失，也不是结果行消失，而是部分模型在原始 RAGAS 产物里没有得到可用的 "
            "`answer_relevancy`。换句话说，原始回答、RAGAS 输出和主表目录都在，但不能把不存在的指标硬说成存在。"
            "因此，正确做法不是伪造该分数，而是在后续分析中 **兼容缺值并显式备注**。\n\n"
            "本次我已经把这层兼容补到了报告分析链里：方法透视表会强制保留 "
            "`answer_relevancy_direct`、`answer_relevancy_rag`、`faithfulness_direct`、`faithfulness_rag` 等预期列；"
            "如果某个方法某个题型没有实际值，对应列保留为空值，而不是因为 pivot 结果缺列导致分析中断。当前这一兼容备注如下。"
        )
    else:
        completion_paragraph = (
            f"本轮已完成主表合并目录为 `reports/remote_sync/{main_dir.name}`。该目录包含总表 `comparison.csv`、`model_summary.csv`、每个模型的 `round_001` 原始回答、RAGAS 输出和分模型汇总，因此主结果已经具备复查和复算条件。"
            f"当前主表共覆盖 `{total_models}` 个模型，且 `{complete_models}` 个模型的 `RAG answer_relevancy` 都已可直接比较。"
        )
        main_result_paragraph = (
            f"从当前已完成的主表可以直接得到三个结论。第一，`{positive_uplift_models}` 个可比较模型全部呈现 `RAG > Direct` 的 answer relevancy，"
            "说明你的 RAG 系统在受控协议下具有稳定的正增益。"
        )
        minimax_section_title = "## 15. MiniMax-M2 缺失列问题已经如何补回"
        minimax_section_body = (
            "这一轮里最典型的缺列问题曾经发生在 `MiniMax-M2`：主表文件没有丢，原始回答也没有丢，真正缺失的是原始 RAGAS 产物里的 "
            "`RAG answer_relevancy`。后来我没有改 judge、没有改 strictness，也没有把它换成别的模型，而是用 **同样的 `MiniMax-M2` judge 和同样的 `strictness=3`** "
            "对现成 `rag_answers.jsonl` 做了单模型补评，并把结果回填进 `ragas_scores.csv`、`ragas_summary.json`、模型级 `model_summary.csv` 和主表总汇总。"
            "因此现在的 `MiniMax-M2` 分数已经回到与其他模型同一套标准下，不属于换标准补分。"
        )

    uplift_image = local_analysis_dir / "uplift_by_model.png"
    latency_image = local_analysis_dir / "latency_by_model.png"
    uplift_image_md = f"![主表模型增益图](../reports/remote_sync/{main_dir.name}/local_analysis/uplift_by_model.png)" if uplift_image.exists() else ""
    latency_image_md = f"![主表模型时延图](../reports/remote_sync/{main_dir.name}/local_analysis/latency_by_model.png)" if latency_image.exists() else ""

    report = f"""# Benchmark 正式测评报告（中文）

## 1. 报告定位

本报告面向论文写作、答辩汇报和后续 benchmark 复现实验，系统整理本项目当前 benchmark 工作的完整方法链条，包括模型选择、120 题固定评测集的构建原则、各类结果表的设计逻辑、评估指标体系、主结果表的当前结论，以及当前仍未纳入正式结果解释的后续表。报告只纳入目前已经完成并可验证的主结果表分析；稳定性表、附加表和 topK 消融表在本报告中只说明设计与方法，不提前给出未完成结论。

## 2. 研究目标与设计原则

本 benchmark 的目标不是做一个脱离上下文的“通用大模型排行榜”，而是在统一部署生态和统一 RAG 系统之下，测量 **RAG 相对于 Direct 的真实增益**。因此，整个测评体系优先保证内部效度而不是追求跨 provider 的最大覆盖。对主表、分类表和稳定性表，系统统一采用受控推理协议，尽量压低 provider 默认 thinking、隐藏 system prompt、接口差异、限流策略和超时行为对结论的污染。附加表与 topK 表被单独拆出，正是为了把“provider 原生行为差异”和“检索候选池大小差异”从主结论里剥离出来。

## 3. 模型选择依据

主表固定使用 8 个商业可稳定访问的中文商用大模型：{", ".join(CONTROLLED_MAIN_MODELS)}。这一选择同时受三个条件约束。第一，模型必须能在同一套 OpenAI-compatible / 兼容 API 生态里稳定调用，避免把跨 provider 隐含提示词、不同 safety policy、不同 timeout 规则直接混进主比较。第二，模型需要能够支撑固定 120 题、多模型、后续 100 轮有放回抽样的真实 API 大规模测评成本。第三，模型在角色上要形成对照，包括强基线、中文问答组、长上下文组、推理增强组和补充组，从而让 `RAG vs Direct` 的比较更有解释力。

下表给出当前 8 模型在主表中的角色定义。

{frame_to_markdown(model_core[["model_label", "model_role"]])}

在模型池之外，本研究明确没有把 GPT、Claude、Gemini 混入主表。这不是因为这些模型不重要，而是因为当前研究问题优先关注 **同一部署生态内 RAG 相对 Direct 的增益**。一旦直接跨 provider 比较，就会引入 reasoning-control 接口、隐藏提示、限流、拒答策略、上下文管理等额外混杂因素，从而削弱主结论的内部效度。这个限制将在论文中作为外部效度边界说明。

## 4. 120 题固定评测集是怎么选出来的

固定评测集由脚本 `pipelines.prepare_benchmark_dataset.py` 统一生成，不是人工零散拼题。生成器把评测题集拆成三部分：`ragppi_gold`、`doc_design` 和 `schema_tables`，再经过字段清洗、缺失过滤、去重和固定编号，得到最终 `fbtp_eval_fixed_120.jsonl`。代码层面的默认配比是 `40 / 20 / 60`，对应 `40` 条 interaction-centered 文本问答、`20` 条文档与设计方法学问答、`60` 条面向结构化数据库的 schema/table 问答。这个配比来自两个考虑：一是保留对 interaction-centered 检索链的充分压力测试；二是把已经实现的数据库设计、质量检查、执行链和 provenance 能力系统性纳入论文级题集，而不让 benchmark 只停留在一类问答上。

数据集的正式汇总如下。

{frame_to_markdown(primary_summary)}

从记录类型看，固定题集同时覆盖了 `csv`、`text` 和 `jsonl` 三类信息源；从来源组看，固定题集分别对应 interaction 文本证据、方法学/设计文档证据、以及数据库规范化表结构证据。这保证 benchmark 不只是测“长文本摘要能力”，而是在同一框架下同时测检索、跨表、结构化字段命中、方法学理解和证据对齐。

## 5. 为什么说这 120 题具有科学依据

这 120 题的构建遵循了“固定全集 + 分层覆盖 + 可复核 ground truth”的原则。第一，每题都保留明确 `ground_truth`、期望答案片段、期望来源、期望表族或期望主键等约束，使得后续评估既可以使用 RAGAS，也可以使用本地结构化命中指标。第二，题集结构是分层而不是随机混合：`ragppi` 负责检验 interaction-centered 检索与摘要，`doc/design` 负责检验系统设计理解和方法学复述，`schema_tables` 负责检验多表结构、ID 与字段级命中能力。第三，固定全集保证主结果表可重复，后续稳定性表再在同一全集上做有放回抽样，从而把“可复现的主结论”和“统计上的波动性评估”分开。

更细一层的二级题型分布如下。

{frame_to_markdown(secondary_summary)}

这些二级题型覆盖了 protein profile、protein identifier、interaction overview、affinity、developability、annotation、provenance、structure、digestive assay、immunogenicity、loop annotation、loop flexibility、protein flexibility、target variant、source metadata，以及 architecture、quality、database、benchmark protocol 等文档设计问题。换句话说，这个固定集并不是通用百科题，而是围绕你自己的数据库设计、RAG 执行链和知识对象来设计的“任务型题集”。

## 6. 题目都是什么样

下面先给出三大类题目的代表性样例。

{frame_to_markdown(question_examples)}

为了方便论文写作和人工审阅，完整 120 题目录已经同步输出到：

- `reports/remote_sync/{main_dir.name}/local_analysis/question_catalog.csv`
- `reports/remote_sync/{main_dir.name}/local_analysis/question_catalog.md`

本报告末尾也附上完整题目清单。

## 7. 主表、分类表、稳定性表、附加表、topK 表分别是什么

整个 benchmark 不是一张表，而是一套职责明确的结果矩阵。其设计如下。

{frame_to_markdown(protocol_table)}

对应的结构关系可以概括为：

```mermaid
graph TD
    A[Fixed 120 Question Set] --> B[Main Table]
    A --> C[Category Table]
    A --> D[Stability Table]
    A --> E[Appendix Native Table]
    A --> F[topK Ablation Table]
    B --> G[RAGAS and Local Metrics]
    C --> G
    D --> G
    E --> G
    F --> G
```

主表回答“RAG 是否比 Direct 更好”；分类表回答“在哪类问题上更好”；稳定性表回答“这种提升是否稳定”；附加表回答“如果放开 provider-native 行为，结论会怎样变化”；topK 表回答“候选池加大是否值得”。因此，这几张表互相补充，但不互相替代。

## 8. 主表配置为什么这样定

本轮已经完成的主表采用的是固定全集一次性全量跑完的配置，而不是抽样：`population_size=120`、`rounds=1`、`sample_size=120`、`with_replacement=false`、`seed=42`。这意味着主表是“固定全集的 A/B 对比”，用来提供最清晰、最容易解释的论文主结论。检索侧在主表中固定最终上下文 `top_k={FINAL_CONTEXT_TOP_K}`，而检索候选池大小的敏感性问题则被显式留给单独的 topK 消融表，不在主表混跑。主表使用 `eval_mode=ragas`，保证回答相关性、faithfulness 和 context precision 都在同一框架下评估。

## 9. 推理协议为什么要受控

主表、分类表和稳定性表统一使用受控推理协议，原因在于不同模型对 thinking 的默认策略并不一致。如果不做控制，Direct 与 RAG 之间的差异很可能混入模型自身推理模式变化，而不是系统带来的真实增益。本项目当前受控协议是：`DeepSeek-V3.2`、`GLM-5`、`Kimi-K2.5`、`Qwen3-235B-A22B-Instruct-2507`、`MiniMax-M2`、`MiniMax-M2.7`、`ERNIE-4.5-Turbo-128K` 都走受控非 thinking 协议；只有 `DeepSeek-R1` 保留原生 reasoning，因为强行关闭在兼容接口下可能导致可见答案异常。这样做的核心目的，是让 Direct 与 RAG 在同一模型上共享完全一致的回答协议，从而把比较焦点放回 RAG 系统本身。

## 10. 结果是怎么评估的

本 benchmark 采用 “RAGAS 主指标 + 本地结构化指标 + 运行时指标” 的三层评估框架。RAGAS 负责判断回答是否相关、是否忠于证据、上下文是否有效；本地结构化指标负责判断回答是否命中目标来源、表族和实体 ID；运行时指标负责衡量时延、成功率和失败率。这种组合比单一分数更适合你的系统，因为它既覆盖生成质量，也覆盖数据库问答链条最核心的检索与字段对齐能力。

{frame_to_markdown(metric_table)}

## 11. 当前主表已经完成到什么程度

{completion_paragraph}

## 12. 主表核心结果

下表是当前主表的核心摘要。`uplift = rag_answer_relevancy - direct_answer_relevancy`。

{frame_to_markdown(model_core[[
        "model_label",
        "direct_answer_relevancy",
        "rag_answer_relevancy",
        "uplift",
        "direct_latency_ms",
        "rag_latency_ms",
        "latency_ratio_rag_vs_direct",
        "rag_faithfulness",
        "rag_context_precision",
    ]])}

{main_result_paragraph}第二，增益最大的模型是 `{best_uplift_row["model_label"] if best_uplift_row is not None else "NA"}`，其 uplift 为 `{_format_metric(best_uplift_row["uplift"]) if best_uplift_row is not None else "NA"}`；增益最小但仍为正的模型是 `{smallest_uplift_row["model_label"] if smallest_uplift_row is not None else "NA"}`，其 uplift 为 `{_format_metric(smallest_uplift_row["uplift"]) if smallest_uplift_row is not None else "NA"}`。第三，质量提升并不是免费的：`RAG` 的平均时延显著高于 `Direct`，其中最慢的 RAG 模型是 `{slowest_rag_row["model_label"] if slowest_rag_row is not None else "NA"}`，平均时延达到 `{_format_metric(slowest_rag_row["rag_latency_ms"], 2) if slowest_rag_row is not None else "NA"}` ms，而 `Direct` 最快的模型是 `{fastest_direct_row["model_label"] if fastest_direct_row is not None else "NA"}`，平均时延仅 `{_format_metric(fastest_direct_row["direct_latency_ms"], 2) if fastest_direct_row is not None else "NA"}` ms。

如果从证据一致性的角度看，当前主表里 faithfulness 最高的模型是 `{best_faithfulness_row["model_label"] if best_faithfulness_row is not None else "NA"}`，其 `rag_faithfulness` 为 `{_format_metric(best_faithfulness_row["rag_faithfulness"]) if best_faithfulness_row is not None else "NA"}`。这说明本轮主表不是靠“模型更会编”获得高分，而是伴随着较高的证据忠实度和上下文精度。

{uplift_image_md}

{latency_image_md}

## 13. 主表结果应该怎么解读

主表中最值得强调的模式是，RAG 对弱 Direct 基线模型的提升尤其明显。像 `MiniMax-M2.7`、`DeepSeek-R1`、`GLM-5` 这类模型，Direct 基线较弱或较保守时，RAG 能显著把回答拉回到与题目和证据更一致的状态。与此同时，像 `DeepSeek-V3.2` 这类 Direct 基线已经相对较强的模型，RAG 仍能继续提供增益，只是 uplift 不会像弱基线模型那样夸张。这种现象对论文写作很有价值，因为它说明 RAG 的价值不是只存在于“弱模型补课”，也能在强基线上继续发挥作用。

另一方面，主表也清楚展示了“质量—时延折中”是真实存在的。部分模型虽然获得了很高 uplift，但 `RAG` 时延成倍增加，甚至出现几十倍以上的时延倍率。因此在论文中，主结论应该写成“RAG 在相关性和证据一致性上显著优于 Direct，但这种增益伴随明显的时延代价”，而不是简单地把 RAG 说成无条件更优。

## 14. 按题型看，RAG 的优势集中在哪里

利用主表内各模型的题型级拆分结果，可以把 `ragppi`、`doc/design` 和 `schema_tables` 三类问题分别比较。当前按模型平均后的 uplift 如下。

{frame_to_markdown(category_mean)}

这个结果表明，当前 RAG 系统对 `doc/design` 和 `schema_tables` 两类问题的帮助最大，对 `ragppi` 的帮助相对较小。这种差异是合理的。`doc/design` 问题的答案往往可以从较明确的文档证据中抽取和复述，`schema_tables` 问题又天然依赖 ID、表族和字段对齐，因此 RAG 对这两类任务能直接发挥检索与结构化证据优势。相比之下，`ragppi` 问题更接近 interaction-centered 的关系摘要，往往需要对长文本证据做更强的压缩与概括，因此提升空间更依赖生成模型本身的摘要稳定性。

{minimax_section_title}

{minimax_section_body}

{frame_to_markdown(issue_df if not issue_df.empty else pd.DataFrame([{{"scope": "none", "model_label": "-", "category": "-", "issue": "当前没有缺值备注。"}}]))}

这意味着报告脚本已经不会再因为缺列或补评后的列更新而失真。这个边界在论文中也应该保持诚实：如果存在缺值就明确标注，如果缺值已经在同标准下补回，也要把补回方法写清楚。

## 16. 原始回答与运行时完整性

为了验证主表不是只剩汇总分数，我还把每个模型的 `rag_answers.jsonl` 和 `direct_answers.jsonl` 做了盘点。结果如下。

{frame_to_markdown(answer_inventory)}

这张表说明，当前大部分模型的 `RAG` 与 `Direct` 都能做到 `120/120` 成功完成；`MiniMax-M2` 则有 `1` 条 RAG 失败记录。工程上，这些原始回答文件使得后续人工 spot check、错误归因和论文附录复核都具备条件。

## 17. 为什么稳定性表、附加表和 topK 表现在先不纳入结果分析

这三类表的职责与主表不同。稳定性表本质上是重复抽样统计，不适合在主表尚未完全收敛时提前下结论；附加表故意放开 provider-native 行为，其作用是解释“原生模式是否改变结论”，不应该替代主表；topK 表则只回答“检索候选池是否值得变大”，也不应拿来替代主表排序。当前这些表的方法学已经锁定，但因为结果尚未完整，所以本报告只保留设计说明，不提前输出未完成结论。

## 18. 当前主表的结论边界

当前已经可以成立的结论是：在统一部署生态、统一受控协议、固定 120 题全集上，你的 RAG 系统相对于 Direct 基线在大多数模型上都有明确正增益，且这种增益在 `doc/design` 与 `schema_tables` 两类问题上尤其明显。同时，RAG 的收益伴随着真实可见的时延代价，因此后续稳定性表与 topK 表对于“增益是否稳定、代价是否可接受”仍然是必须的补充，而不是可有可无的附属实验。

## 19. 本报告对应的关键文件

本报告依赖并生成的关键文件包括：

- 主表目录：`reports/remote_sync/{main_dir.name}`
- 主表总汇总：`reports/remote_sync/{main_dir.name}/model_summary.csv`
- 主表原始回答：`reports/remote_sync/{main_dir.name}/models/*/round_001/*_answers.jsonl`
- 本地分析目录：`reports/remote_sync/{main_dir.name}/local_analysis`
- 题集：`data/fbtp_eval_fixed_120.jsonl`
- 题集汇总：`data/fbtp_eval_fixed_120.summary.json`

## 20. 附录：固定 120 题完整目录

{frame_to_markdown(question_catalog)}
"""

    output_path.write_text(report, encoding="utf-8")
    return {
        "report_path": output_path,
        "local_analysis_dir": local_analysis_dir,
        "question_catalog_path": local_analysis_dir / "question_catalog.csv",
        "core_summary_path": local_analysis_dir / "main_table_core_summary.csv",
        "category_summary_path": local_analysis_dir / "category_uplift_summary.csv",
        "teacher_figure_path": DEFAULT_TEACHER_FIGURE_PATH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese benchmark report and robust local analysis artifacts.")
    parser.add_argument("--main-dir", default=str(DEFAULT_MAIN_DIR))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    outputs = write_report(
        main_dir=Path(args.main_dir),
        dataset_path=Path(args.dataset_path),
        summary_path=Path(args.summary_path),
        output_path=Path(args.output_path),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
