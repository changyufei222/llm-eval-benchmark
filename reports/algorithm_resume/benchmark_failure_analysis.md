# Benchmark Failure Analysis

## 1. Why a failure analysis matters here

The benchmark already has strong final results, but algorithm interviewers will still ask:

- where RAG helps the most
- where RAG helps the least
- which tradeoffs remain expensive
- which failure patterns forced protocol cleanup

This note summarizes the main failure and limitation patterns exposed by the promoted benchmark release.

## 2. Failure / weakness taxonomy

For this benchmark, the most useful taxonomy is:

1. `Low Direct baseline`
   - the model is weak even before retrieval
2. `Small RAG uplift`
   - retrieval helps, but only marginally
3. `Category-specific weakness`
   - one question family benefits less than others
4. `Latency explosion`
   - RAG quality improves, but cost/latency grows sharply
5. `Provider-native divergence`
   - controlled protocol and provider-native appendix differ
6. `Replay instability / quota contamination`
   - intermediate runs contain retry, failed rows, or provider artifacts and cannot be promoted as official conclusions

## 3. Main table failure patterns

### Pattern A: Direct baseline is near zero, so uplift is large but fragile

Examples:

- `MiniMax-M2.7`: `Direct = 0.0056`, `RAG = 0.9072`, uplift `+0.9016`
- `MiniMax-M2`: `Direct = 0.0139`, `RAG = 0.7701`, uplift `+0.7562`

Interpretation:

- these are excellent RAG uplift stories
- but they also show that the non-RAG baseline is extremely weak
- in interviews, this should be framed as:
  - retrieval is essential for these models on this task
  - not as “the model is universally strong”

### Pattern B: some models already have a strong Direct baseline, so RAG uplift is smaller

Example:

- `DeepSeek-V3.2`: `Direct = 0.6398`, `RAG = 0.7660`, uplift `+0.1262`

Interpretation:

- smaller uplift does **not** mean retrieval is useless
- it means the model’s own baseline answer capability is already relatively strong
- the correct reading is:
  - RAG still helps
  - but the marginal gain is smaller once the Direct baseline is high

## 4. Category-specific patterns

From the promoted category summary:

- `doc/design`: mean uplift `0.7033`
- `schema_tables`: mean uplift `0.5884`
- `ragppi`: mean uplift `0.4079`

Interpretation:

- document/design and schema-structured questions benefit most from retrieval
- interaction-style `ragppi` questions still improve, but the average uplift is lower

Important hard case:

- `ERNIE-4.5-Turbo-128K` on `ragppi`: uplift is slightly negative (`-0.0108`)

This is a good example to discuss because it shows:

- RAG is not magically better on every model-category pair
- the benchmark can surface negative or near-zero gains instead of hiding them

## 5. topK failure patterns

The topK ablation is one of the strongest algorithm-facing parts of this repo because it shows a non-trivial tradeoff:

- `topK = 32`: mean answer relevancy `0.7614`, latency `45138.84 ms`
- `topK = 64`: mean answer relevancy `0.7480`, latency `77357.43 ms`
- `topK = 128`: mean answer relevancy `0.7146`, latency `143530.31 ms`

Interpretation:

- bigger candidate pools do not monotonically improve answer quality
- they increase latency very aggressively
- the failure mode is therefore not “retrieval misses because k is too small”
- the more realistic failure mode is:
  - noisy candidate expansion
  - slower downstream reasoning
  - worse overall quality-efficiency balance

This is exactly the kind of ablation result that reads as algorithmic evaluation, not just engineering.

## 6. Stability and cleanup patterns

The repo historically contained:

- quota issues
- retry traces
- repair scripts
- backup folders

The promoted release explicitly avoids treating those noisy runs as the official benchmark conclusion.

Instead, the clean promoted stability release uses:

- `2` models
- `100` rounds each
- `RAG-only`
- `failed_rows_total = 0`

This matters because one of the main benchmark “failure patterns” was originally not model quality, but **provider/runtime contamination**.  

The clean stability release is the answer to that problem.

## 7. Representative benchmark lessons

### Lesson 1: RAG can help all models, but not equally

The benchmark does not claim one universal uplift size.

It shows:

- all `8/8` models improve on the main table
- but the uplift ranges from `+0.1262` to `+0.9016`

### Lesson 2: category structure matters

Retrieval is especially valuable when:

- evidence is document-heavy
- evidence is schema-structured
- the answer depends on retrieved context rather than free prior knowledge

### Lesson 3: bigger retrieval is not always better

This is one of the most interview-useful findings:

- `topK` growth increases cost dramatically
- quality can plateau or even worsen

### Lesson 4: a trustworthy benchmark must separate official results from repair history

This repo now does that explicitly.

That is a benchmark-design strength, not just repo cleanup.

## 8. Interview-ready explanation

A concise explanation you can use:

> The benchmark was useful not only because it showed that RAG usually helps, but because it exposed *where* it helps, *how much* it helps, and *what it costs*.  
> The strongest patterns were category-specific uplift, non-monotonic top-k behavior, and the need to separate clean promoted replay results from earlier quota- or retry-contaminated runs.
