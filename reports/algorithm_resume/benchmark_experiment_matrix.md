# Benchmark Experiment Matrix

This file compresses the benchmark into one page that emphasizes evaluation design rather than engineering scaffolding.

## Experiment Matrix

| Component | Scope | Models / Conditions | Main Metric | Headline Result |
|---|---|---|---|---|
| Main table | fixed `120` questions, Direct vs RAG | `8` models | answer relevancy uplift | all `8/8` models show positive `RAG - Direct` uplift |
| Category table | `doc/design`, `ragppi`, `schema_tables` | `8 x 3` buckets | category uplift | retrieval helps most on `doc/design` and `schema_tables` |
| Appendix table | provider-native supplementary protocol | same `8` models | appendix uplift | supplementary protocol exposes provider-native behavior differences |
| topK ablation | `candidate_top_k ∈ {32,64,128}` | same `8` models | quality-efficiency tradeoff | `topK = 32` is the best promoted overall tradeoff |
| Stability release | RAG-only repeated replay | `DeepSeek-V3.2` vs `MiniMax-M2.7`, `100` rounds each | answer relevancy mean and `95% CI` | `MiniMax-M2.7` leads on quality, `DeepSeek-V3.2` leads on latency |

## Main Table Snapshot

| Model | Direct | RAG | Uplift | Latency Ratio |
|---|---:|---:|---:|---:|
| MiniMax-M2.7 | 0.0056 | 0.9072 | +0.9016 | 29.91x |
| MiniMax-M2 | 0.0139 | 0.7701 | +0.7562 | 99.26x |
| DeepSeek-R1 | 0.0682 | 0.7326 | +0.6644 | 3.67x |
| GLM-5 | 0.1391 | 0.7772 | +0.6381 | 3.90x |
| Qwen3-235B-A22B-Instruct-2507 | 0.1795 | 0.7388 | +0.5593 | 91.16x |
| Kimi-K2.5 | 0.1649 | 0.6350 | +0.4701 | 7.04x |
| ERNIE-4.5-Turbo-128K | 0.4061 | 0.7309 | +0.3248 | 30.71x |
| DeepSeek-V3.2 | 0.6398 | 0.7660 | +0.1262 | 5.61x |

## Category Snapshot

| Category | Mean Uplift | Median Uplift |
|---|---:|---:|
| doc/design | 0.7033 | 0.76735 |
| schema_tables | 0.5884 | 0.65505 |
| ragppi | 0.4079 | 0.3726 |

## topK Snapshot

| candidate_top_k | Mean RAG Answer Relevancy | Mean Uplift | Mean RAG Latency ms |
|---|---:|---:|---:|
| 32 | 0.7614 | +0.5593 | 45138.84 |
| 64 | 0.7480 | +0.5459 | 77357.43 |
| 128 | 0.7146 | +0.4856 | 143530.31 |

## Stability Snapshot

| Model | Rounds | Failed Rows Total | Answer Relevancy Mean | 95% CI | Latency Mean ms |
|---|---:|---:|---:|---|---:|
| DeepSeek-V3.2 | 100 | 0 | 0.7624 | [0.7532, 0.7716] | 11009.04 |
| MiniMax-M2.7 | 100 | 0 | 0.8220 | [0.8085, 0.8356] | 24610.60 |

## Interview Readout

- This repo is not just “a benchmark system”; it is a controlled multi-part evaluation protocol.
- The key algorithm-facing skill is the separation of:
  - headline quality comparison
  - question-category sensitivity
  - retrieval hyperparameter sensitivity
  - repeated-run stability
- The strongest summary sentence is:
  - designed and froze a fixed-set benchmark protocol for Direct-vs-RAG comparison, with category analysis, top-k ablation, and repeated-run stability evaluation across multiple large models.
