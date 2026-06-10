# Benchmark Protocol Summary

## 1. What this benchmark is actually testing

This project is not mainly about model training.  
Its algorithm-facing value is that it defines a **controlled evaluation protocol** for testing whether retrieval augmentation improves answer quality in a grounded FBBP setting.

The benchmark is designed to separate:

- answer quality under `Direct` vs `RAG`
- question-type sensitivity
- retrieval hyperparameter sensitivity
- repeated-run stability
- latency-quality tradeoffs

## 2. Frozen benchmark scope

The promoted official benchmark uses a fixed benchmark package:

- dataset: `fixed 120`
- comparison: `Direct` vs `RAG`
- models: `8`

The official promoted result scope is:

1. main table
2. category table
3. appendix table
4. topK ablation
5. stability release

## 3. Main table protocol

Main table definition:

- same fixed `120` questions for every model
- same `Direct / RAG` protocol for every model
- same summary metrics across all `8` models

The main table answers the central question:

> Does RAG improve answer relevancy compared with Direct answering under a controlled protocol?

Promoted result:

- all `8/8` models show positive `RAG - Direct` uplift on answer relevancy

## 4. Category table protocol

The benchmark does not stop at one global mean.  
It also splits the fixed set into category buckets:

- `ragppi`
- `doc/design`
- `schema_tables`

This is important because a RAG system may help some categories more than others.

The category table therefore answers:

> On which problem types does retrieval produce the largest gain?

Promoted result:

- `doc/design` and `schema_tables` show the strongest mean uplift

## 5. topK ablation protocol

The benchmark also isolates retrieval hyperparameters:

- `candidate_top_k ∈ {32, 64, 128}`

This is not mixed into the main table.  
It is run as a separate ablation because it answers a different question:

> How sensitive is the RAG system to candidate pool size, and what is the quality-efficiency tradeoff?

Promoted result:

- `topK = 32` is the best overall promoted tradeoff

Interpretation:

- larger candidate pools increase latency sharply
- larger candidate pools do not guarantee better quality

## 6. Stability protocol

The stability release is a repeated-run clean replay:

- models: `DeepSeek-V3.2`, `MiniMax-M2.7`
- mode: `RAG-only`
- rounds per model: `100`
- outputs:
  - answer relevancy mean
  - `95% CI`
  - latency mean / p50 / p95
  - failed rows total
  - full success rounds

This answers:

> Are the promoted conclusions stable under repeated local replay, or are they dominated by one-off provider noise?

Promoted result:

- both promoted stability models have `failed_rows_total = 0`
- `MiniMax-M2.7` leads on quality
- `DeepSeek-V3.2` leads on latency

## 7. Why this is stronger than “just running a benchmark”

The protocol has three strengths:

1. **frozen fixed-set comparison**
   - avoids drifting benchmark scope
2. **dimension separation**
   - main quality table
   - category sensitivity
   - retrieval ablation
   - stability replay
3. **promotion rules**
   - official artifacts are separated from smoke outputs, repair logs, and backup folders

This makes the repo look much more like:

- benchmark design
- experimental protocol design
- evaluation methodology

rather than only:

- “I ran some scripts and plotted some tables.”

## 8. Recommended algorithm-facing framing

The strongest framing is:

> Designed and froze a fixed-set benchmark protocol for Direct-vs-RAG comparison across 8 models, explicitly separating global quality, category sensitivity, retrieval top-k ablation, and repeated-run stability to make conclusions interpretable and reproducible.
