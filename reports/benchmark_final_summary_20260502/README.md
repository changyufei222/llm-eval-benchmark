# Benchmark Final Summary Package

This directory is the canonical promoted result package for `llm-eval-benchmark`.

## Use This Directory For

- Final benchmark conclusions
- Portfolio or interview summaries
- Teacher-facing benchmark updates
- Manuscript or appendix drafting

## Core Files

- `benchmark_results_final_summary_cn.md`
- `benchmark_results_final_summary.md`
- `benchmark_master_overview.csv`
- `main_table_ranked_summary.csv`
- `category_overall_uplift_summary.csv`
- `category_model_uplift_detail.csv`
- `appendix_ranked_summary.csv`
- `topk_overall_summary.csv`
- `topk_detail_by_model.csv`
- `topk_best_k_by_model.csv`
- `stability_final_summary.csv`

## Interpretation Shortcuts

- Main table: use for the headline `RAG vs Direct` conclusion on the fixed-120 benchmark
- Category table: use for which task families benefit most from RAG
- Appendix table: use for provider-native supplementary comparison
- `topK` table: use only for retrieval sensitivity and quality-latency tradeoff
- Stability table: use for the clean quality-versus-latency comparison between `MiniMax-M2.7` and `DeepSeek-V3.2`

## Source Tables

`source_tables/` contains the upstream summary tables that were consolidated into this final package:

- `category_summary.csv`
- `appendix_model_summary.csv`
- `topk_032_model_summary.csv`
- `topk_064_model_summary.csv`
- `topk_128_model_summary.csv`

## Integrity Note

This package is the promoted release surface. Do not replace it with `reports/latest/`, smoke folders, backup folders, or raw repair traces when writing final conclusions.
