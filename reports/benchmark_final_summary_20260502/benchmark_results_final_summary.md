# Benchmark Results Final Summary

## 1. Scope and file organization

- This folder consolidates the final usable benchmark outputs across main, category, appendix, topK, and stability tables.
- All table-specific summary CSV files are placed beside this report for direct reuse in manuscript/report writing.

## 2. Main table

- Scope: 8 models, Direct vs RAG, fixed 120-question benchmark.
- Core file: `main_table_ranked_summary.csv`
- Final interpretation: every model benefits from RAG on answer relevancy, but the uplift and latency cost differ substantially by model family.

- MiniMax-M2.7: Direct 0.0056, RAG 0.9072, uplift +0.9016, latency ratio 29.91x
- MiniMax-M2: Direct 0.0139, RAG 0.7701, uplift +0.7562, latency ratio 99.26x
- DeepSeek-R1: Direct 0.0682, RAG 0.7326, uplift +0.6644, latency ratio 3.67x
- GLM-5: Direct 0.1391, RAG 0.7772, uplift +0.6381, latency ratio 3.90x
- Qwen3-235B-A22B-Instruct-2507: Direct 0.1795, RAG 0.7388, uplift +0.5593, latency ratio 91.16x
- Kimi-K2.5: Direct 0.1649, RAG 0.6350, uplift +0.4701, latency ratio 7.04x
- ERNIE-4.5-Turbo-128K: Direct 0.4061, RAG 0.7309, uplift +0.3248, latency ratio 30.71x
- DeepSeek-V3.2: Direct 0.6398, RAG 0.7660, uplift +0.1262, latency ratio 5.61x

## 3. Category table

- Scope: question-type breakdown across `doc/design`, `ragppi`, and `schema_tables`.
- Core file: `category_overall_uplift_summary.csv`
- Final interpretation: category-level gains are not uniform; structural/document-design tasks benefit the most from retrieval augmentation.

- doc/design: mean uplift 0.7033, median uplift 0.7673
- schema_tables: mean uplift 0.5884, median uplift 0.6550
- ragppi: mean uplift 0.4079, median uplift 0.3726

## 4. Appendix table

- Scope: supplementary 8-model native-thinking/provider-behavior table.
- Core file: `appendix_ranked_summary.csv`
- Final interpretation: appendix results provide a parallel view of model behavior under the supplementary protocol and should be read as support evidence rather than replacing the main table.

- MiniMax-M2.7: Direct 0.2148, RAG 0.9376, uplift +0.7228, appendix-main RAG +0.0304
- Qwen3-235B-A22B-Instruct-2507: Direct 0.1683, RAG 0.7231, uplift +0.5548, appendix-main RAG -0.0157
- GLM-5: Direct 0.2510, RAG 0.7416, uplift +0.4906, appendix-main RAG -0.0356
- Kimi-K2.5: Direct 0.2390, RAG 0.6899, uplift +0.4509, appendix-main RAG +0.0549
- ERNIE-4.5-Turbo-128K: Direct 0.3922, RAG 0.7437, uplift +0.3515, appendix-main RAG +0.0128
- MiniMax-M2: Direct 0.2412, RAG 0.4948, uplift +0.2536, appendix-main RAG -0.2753
- DeepSeek-V3.2: Direct 0.5052, RAG 0.7399, uplift +0.2347, appendix-main RAG -0.0261
- DeepSeek-R1: Direct 0.5971, RAG 0.7332, uplift +0.1361, appendix-main RAG +0.0006

## 5. TopK table

- Scope: official topK ablation uses `candidate_top_k ∈ {32, 64, 128}`. `256` is deprecated and excluded from formal interpretation.
- Core files: `topk_overall_summary.csv`, `topk_detail_by_model.csv`, `topk_best_k_by_model.csv`
- Final interpretation: larger candidate pools do not monotonically improve quality; they raise latency sharply, so topK should be discussed as a quality-efficiency tradeoff rather than “bigger is better”.

- topK=32: mean RAG answer relevancy 0.7614, mean uplift +0.5593, mean RAG latency 45138.84 ms
- topK=64: mean RAG answer relevancy 0.7480, mean uplift +0.5459, mean RAG latency 77357.43 ms
- topK=128: mean RAG answer relevancy 0.7146, mean uplift +0.4856, mean RAG latency 143530.31 ms

## 6. Stability table

- Scope: final clean RAG-only stability comparison on 2 models (`DeepSeek-V3.2`, `MiniMax-M2.7`), each with 100 rounds.
- Core file: `stability_final_summary.csv`
- Integrity check: all 200 rounds are present and all rounds have `failed_rows = 0`.
- Final interpretation: `MiniMax-M2.7` has higher average answer quality, while `DeepSeek-V3.2` is significantly faster.

- DeepSeek-V3.2: answer relevancy 0.7624, 95% CI [0.7532, 0.7716], latency mean 11009.04 ms
- MiniMax-M2.7: answer relevancy 0.8220, 95% CI [0.8085, 0.8356], latency mean 24610.60 ms

## 7. Overall benchmark conclusion

- Main table establishes the core result: retrieval augmentation improves answer relevancy across the 8-model benchmark, but quality gains must be interpreted together with latency cost.
- Category table refines the story: retrieval helps most on document/design and schema-structured questions.
- Appendix table shows supplementary protocol behavior, useful for discussion and robustness framing.
- TopK table shows that candidate-pool expansion has diminishing returns and strong latency penalties, so the best setting should be chosen by use case rather than by quality alone.
- Stability table now supports a clean final comparison: MiniMax-M2.7 is quality-leading, whereas DeepSeek-V3.2 is much faster and still strong in quality.