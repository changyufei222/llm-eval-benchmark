# Benchmark Reliability Note

## 1. Why this note exists

Strong benchmark numbers are not persuasive by themselves unless the protocol is trustworthy.  

For this benchmark, the main reliability questions are:

- is the dataset frozen
- is the official result scope clearly separated from smoke outputs
- were noisy provider / quota / repair traces filtered out of the headline conclusion
- are stability claims backed by repeated replay rather than one-off best-case runs

## 2. Frozen benchmark surface

The promoted official benchmark is explicitly tied to:

- fixed-set dataset:
  - `data/fbtp_eval_fixed_120.jsonl`
- official result directory:
  - `reports/benchmark_final_summary_20260502`
- official release gate:
  - `FINAL_RELEASE_GATE.md`

This matters because it means the result is not supposed to drift with every local smoke run.

## 3. Separation between official and non-official outputs

The release gate explicitly distinguishes:

### Official

- fixed-120 main table
- category table
- appendix table
- topK ablation
- clean selected-RAG stability release

### Not official

- `reports/latest/*`
- 24-row smoke slice
- backup folders
- quota / retry repair traces
- temporary logs

This is one of the strongest signs that the repo is not just “collecting all runs and cherry-picking later”.

## 4. Why the promoted benchmark is more credible than the older smoke checkpoint

The older smoke checkpoint was useful for engineering validation, but not strong enough as the final external benchmark story.

The promoted release improves credibility because it adds:

- a fixed `120`-question official set
- a controlled `8-model` comparison table
- category-specific breakdown
- topK ablation
- a clean repeated-run stability release with `100` rounds per model

So the official story moved from:

- “we have some benchmark outputs”

to:

- “we have a frozen benchmark protocol with official artifacts and release boundaries”

## 5. Why the stability story is credible

The promoted stability release reports:

- `DeepSeek-V3.2`: `100` rounds, `failed_rows_total = 0`
- `MiniMax-M2.7`: `100` rounds, `failed_rows_total = 0`
- `95% CI` reported for answer relevancy

This is much stronger than relying on:

- one lucky run
- a single summary without replay
- a mixed folder containing both clean and repaired results

The repo explicitly uses a **clean local sync** path as the canonical stability source.

## 6. Why the benchmark is not “just a dashboard”

The final summary and release gate explicitly say:

- the dashboard is presentation-only
- it is not the benchmark release gate

That distinction matters for credibility:

- official conclusions come from frozen CSV/summary artifacts
- not from presentation layers

## 7. What the benchmark still does not claim

Even with the promoted release, the correct interpretation remains bounded:

- it demonstrates strong Direct-vs-RAG uplift on the fixed-120 benchmark
- it demonstrates category and topK sensitivity under the defined protocol
- it demonstrates clean repeated-run stability for the promoted stability models

It does **not** automatically prove:

- universal performance on any future dataset
- immunity to future provider-side drift
- that all possible models or prompt settings have been exhausted

This boundary should be stated explicitly in interviews.

## 8. Recommended reliability wording

A strong and safe way to describe the repo is:

> I froze the benchmark surface into an official fixed-120 release, separated official artifacts from smoke and repair history, and promoted only the results that passed a clean release-gate boundary. The final stability conclusion is based on 100-round replay with zero failed rows, not on a single best-case run.
