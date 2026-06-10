# Benchmark 结果总汇总（最终版）

## 一、整理范围

- 本目录汇总了当前所有正式使用的 benchmark 结果：主表、分类表、附加表、topK 表、稳定性表。
- 每张表都对应一个可直接复用的核心 CSV，方便后续写周报、总报告、论文正文或附录。

## 二、主表（Main Table）

- 范围：`8 模型 × Direct/RAG × 固定120题`
- 核心文件：`main_table_ranked_summary.csv`
- 整体结论：8 个模型全部出现 `RAG > Direct` 的回答相关性提升，但提升幅度和时延代价差异很大。

- MiniMax-M2.7：Direct 0.0056，RAG 0.9072，提升 +0.9016，时延倍率 29.91x
- MiniMax-M2：Direct 0.0139，RAG 0.7701，提升 +0.7562，时延倍率 99.26x
- DeepSeek-R1：Direct 0.0682，RAG 0.7326，提升 +0.6644，时延倍率 3.67x
- GLM-5：Direct 0.1391，RAG 0.7772，提升 +0.6381，时延倍率 3.90x
- Qwen3-235B-A22B-Instruct-2507：Direct 0.1795，RAG 0.7388，提升 +0.5593，时延倍率 91.16x
- Kimi-K2.5：Direct 0.1649，RAG 0.6350，提升 +0.4701，时延倍率 7.04x
- ERNIE-4.5-Turbo-128K：Direct 0.4061，RAG 0.7309，提升 +0.3248，时延倍率 30.71x
- DeepSeek-V3.2：Direct 0.6398，RAG 0.7660，提升 +0.1262，时延倍率 5.61x

## 三、分类表（Category Table）

- 范围：按 `doc/design`、`ragppi`、`schema_tables` 三类题进行拆分。
- 核心文件：`category_overall_uplift_summary.csv`、`category_model_uplift_detail.csv`
- 整体结论：RAG 对不同题型的帮助不均衡，其中 `doc/design` 与 `schema_tables` 的平均提升更明显。

- doc/design：平均提升 0.7033，中位提升 0.7673
- schema_tables：平均提升 0.5884，中位提升 0.6550
- ragppi：平均提升 0.4079，中位提升 0.3726

## 四、附加表（Appendix Table）

- 范围：`8 模型` 的 supplementary / native-thinking 补充表。
- 核心文件：`appendix_ranked_summary.csv`
- 整体结论：附加表用于补充观察模型在附加协议下的行为差异，不替代主表，但可以支撑讨论部分的补强证据。

- MiniMax-M2.7：Direct 0.2148，RAG 0.9376，提升 +0.7228，相对主表 RAG 差值 +0.0304
- Qwen3-235B-A22B-Instruct-2507：Direct 0.1683，RAG 0.7231，提升 +0.5548，相对主表 RAG 差值 -0.0157
- GLM-5：Direct 0.2510，RAG 0.7416，提升 +0.4906，相对主表 RAG 差值 -0.0356
- Kimi-K2.5：Direct 0.2390，RAG 0.6899，提升 +0.4509，相对主表 RAG 差值 +0.0549
- ERNIE-4.5-Turbo-128K：Direct 0.3922，RAG 0.7437，提升 +0.3515，相对主表 RAG 差值 +0.0128
- MiniMax-M2：Direct 0.2412，RAG 0.4948，提升 +0.2536，相对主表 RAG 差值 -0.2753
- DeepSeek-V3.2：Direct 0.5052，RAG 0.7399，提升 +0.2347，相对主表 RAG 差值 -0.0261
- DeepSeek-R1：Direct 0.5971，RAG 0.7332，提升 +0.1361，相对主表 RAG 差值 +0.0006

## 五、TopK 表

- 范围：正式协议仅保留 `candidate_top_k = 32 / 64 / 128`。
- `256` 已废弃，不进入正式结果口径。
- 核心文件：`topk_overall_summary.csv`、`topk_detail_by_model.csv`、`topk_best_k_by_model.csv`
- 整体结论：候选池增大并不是单调增益；质量有波动，但时延会显著升高，因此应按质量-效率权衡来解读。

- topK=32：平均 RAG answer_relevancy 0.7614，平均提升 +0.5593，平均 RAG 时延 45138.84 ms
- topK=64：平均 RAG answer_relevancy 0.7480，平均提升 +0.5459，平均 RAG 时延 77357.43 ms
- topK=128：平均 RAG answer_relevancy 0.7146，平均提升 +0.4856，平均 RAG 时延 143530.31 ms

## 六、稳定性表（Stability Table）

- 范围：`2 模型 × 100 轮 × RAG-only` 的 clean 稳定性表。
- 核心文件：`stability_final_summary.csv`
- 完整性校验：`200` 个轮次全部存在，且所有轮次 `failed_rows = 0`。
- 整体结论：`MiniMax-M2.7` 平均质量更高，`DeepSeek-V3.2` 延迟更低，二者构成清晰的质量-效率权衡。

- DeepSeek-V3.2：answer_relevancy 0.7624，95% CI [0.7532, 0.7716]，平均延迟 11009.04 ms
- MiniMax-M2.7：answer_relevancy 0.8220，95% CI [0.8085, 0.8356]，平均延迟 24610.60 ms

## 七、总的 benchmark 结论

- 主表给出主结论：RAG 在 8 模型主 benchmark 中普遍提升回答相关性，但代价是不同程度的时延增长。
- 分类表进一步说明：RAG 的价值在 `doc/design` 与 `schema_tables` 任务上最突出。
- 附加表说明：补充协议下的模型行为与主表并不完全一致，可作为讨论部分的补强证据。
- topK 表说明：候选池扩大存在收益递减，不能简单理解为越大越好。
- 稳定性表说明：在 clean 条件下，`MiniMax-M2.7` 是质量领先者，`DeepSeek-V3.2` 是效率更优者。