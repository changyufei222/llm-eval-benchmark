# Final Result Summary

Promoted on `2026-05-02`.

This file is the canonical portfolio summary for `llm-eval-benchmark`. The official benchmark conclusion is no longer the older 24-row smoke checkpoint. The promoted release is the fixed-120 benchmark package plus the clean selected-RAG stability release.

## Positioning

`llm-eval-benchmark` is an evaluation harness for measuring grounded answer quality, retrieval alignment, and runtime tradeoffs of this project's RAG system versus Direct answering under a controlled multi-model protocol.

## Official Release Status

- Status: promoted final benchmark release
- Release gate: `FINAL_RELEASE_GATE.md`
- Official fixed-set dataset: `data/fbtp_eval_fixed_120.jsonl`
- Official summary directory: `reports/benchmark_final_summary_20260502`
- Chinese final summary: `reports/benchmark_final_summary_20260502/benchmark_results_final_summary_cn.md`
- English final summary: `reports/benchmark_final_summary_20260502/benchmark_results_final_summary.md`

## Official Result Scope

- Main table: `8 models x Direct/RAG x fixed 120 questions`
- Category table: `8 models x {ragppi, doc/design, schema_tables}`
- Appendix table: `8 models x provider-native supplementary mode`
- `topK` ablation: `8 models x candidate_top_k {32, 64, 128}`
- Stability release: `DeepSeek-V3.2` vs `MiniMax-M2.7`, `100` rounds each, `RAG-only`, clean local sync with `failed_rows_total = 0`

## Headline Findings

- Main table: all `8/8` models show positive `RAG - Direct` answer relevancy uplift on the fixed-120 set.
- Strongest main-table uplift: `MiniMax-M2.7`, from `0.0056` to `0.9072`, uplift `+0.9016`.
- Best promoted overall `topK` tradeoff: `candidate_top_k = 32`.
- Clean stability release:
  - `MiniMax-M2.7`: `answer_relevancy = 0.8220`, `95% CI [0.8085, 0.8356]`, `failed_rows_total = 0`
  - `DeepSeek-V3.2`: `answer_relevancy = 0.7624`, `95% CI [0.7532, 0.7716]`, `failed_rows_total = 0`
- Stability interpretation: `MiniMax-M2.7` leads on quality; `DeepSeek-V3.2` leads on latency.

## Canonical Artifacts

- `FINAL_RELEASE_GATE.md`
- `reports/benchmark_final_summary_20260502/benchmark_master_overview.csv`
- `reports/benchmark_final_summary_20260502/main_table_ranked_summary.csv`
- `reports/benchmark_final_summary_20260502/category_overall_uplift_summary.csv`
- `reports/benchmark_final_summary_20260502/appendix_ranked_summary.csv`
- `reports/benchmark_final_summary_20260502/topk_overall_summary.csv`
- `reports/benchmark_final_summary_20260502/topk_best_k_by_model.csv`
- `reports/benchmark_final_summary_20260502/stability_final_summary.csv`
- `reports/remote_sync/stability_selected_rag_best_vs_worst_20260423_133411/local_analysis_final_clean/completion_check.json`

## Not Official

- `reports/latest/*` local smoke and reproducibility outputs
- `data/fbtp_eval.jsonl` 24-row runnable smoke slice
- `reports/remote_sync/*backup*`
- quota / retry / rate-limit repair logs and temporary rerun folders
- `docs/benchmark_formal_report_cn_2026-04-19.md` historical snapshot
- dashboard or control-plane UI as a release gate
