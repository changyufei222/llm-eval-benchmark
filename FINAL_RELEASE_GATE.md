# Final Release Gate

Release date: `2026-05-02`  
Status: `PASSED`

This file defines the only official release surface for `llm-eval-benchmark`. Use it to separate promoted benchmark conclusions from smoke runs, repair logs, backup folders, and historical intermediate reports.

## Canonical Entry Points

- Root summary: `FINAL_RESULT_SUMMARY.md`
- Formal protocol note: `docs/benchmark_protocol_cn.md`
- Reports structure note: `reports/README_RELEASE_STRUCTURE.md`
- Official result directory: `reports/benchmark_final_summary_20260502`
- Chinese final summary: `reports/benchmark_final_summary_20260502/benchmark_results_final_summary_cn.md`
- English final summary: `reports/benchmark_final_summary_20260502/benchmark_results_final_summary.md`

## Optional Showcase Layer

- `docs/benchmark_showcase.html`

This page is presentation-only. It is meant to make the benchmark easier to show, not to replace the official result files above.

## Release Scope

- Main table: `8 models x Direct/RAG x fixed 120 questions`
- Category table: `8 models x {ragppi, doc/design, schema_tables}`
- Appendix table: `8 models x provider-native supplementary mode`
- `topK` ablation: `8 models x candidate_top_k {32, 64, 128}`
- Stability release: `DeepSeek-V3.2` vs `MiniMax-M2.7`, `100` rounds each, `RAG-only`, clean local sync

## Promotion Checklist

- [x] The root summary is promoted away from the old 24-row smoke checkpoint.
- [x] The promoted benchmark package is consolidated under `reports/benchmark_final_summary_20260502`.
- [x] Main, category, appendix, `topK`, and stability tables each have canonical summary CSV files.
- [x] The promoted stability release has a clean verification trail:
  - `reports/remote_sync/stability_selected_rag_best_vs_worst_20260423_133411/local_analysis_final_clean/completion_check.json`
- [x] Historical repair artifacts are separated from official results and kept only for auditability.
- [x] External-facing docs now distinguish official results from smoke or temporary outputs.

## Official Artifacts

- `data/fbtp_eval_fixed_120.jsonl`
- `reports/benchmark_final_summary_20260502/benchmark_master_overview.csv`
- `reports/benchmark_final_summary_20260502/main_table_ranked_summary.csv`
- `reports/benchmark_final_summary_20260502/category_overall_uplift_summary.csv`
- `reports/benchmark_final_summary_20260502/category_model_uplift_detail.csv`
- `reports/benchmark_final_summary_20260502/appendix_ranked_summary.csv`
- `reports/benchmark_final_summary_20260502/topk_overall_summary.csv`
- `reports/benchmark_final_summary_20260502/topk_detail_by_model.csv`
- `reports/benchmark_final_summary_20260502/topk_best_k_by_model.csv`
- `reports/benchmark_final_summary_20260502/stability_final_summary.csv`

## Audit-Only Or Historical Artifacts

- `reports/latest/*`
- `data/fbtp_eval.jsonl`
- `reports/remote_sync/*backup*`
- one-off repair logs, rate-limit retries, and quota recovery traces
- `docs/benchmark_formal_report_cn_2026-04-19.md`
- dashboard and control-plane inspection surfaces

## External Narrative Guardrails

- Cite the fixed-120 main table for the headline benchmark conclusion.
- Cite the selected-RAG clean stability release for the quality-versus-latency tradeoff.
- Cite `topK` only as retrieval-sensitivity analysis, not as a replacement for the main table.
- Do not mix repair history, provider quota incidents, or backup folders into the benchmark headline.
- Treat the dashboard as an engineering support surface, not as the product release gate for this repo.

## Final Headline

- All `8/8` models in the main table show positive `RAG - Direct` answer relevancy uplift.
- `candidate_top_k = 32` is the best promoted overall tradeoff among `32 / 64 / 128`.
- The clean stability release shows `MiniMax-M2.7` leading on quality and `DeepSeek-V3.2` leading on latency.
