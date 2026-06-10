# FBBP 评测算法报告

## 结论摘要

这是一条面向算法解释的 **领域评测协议设计** 线。它不是单纯跑一堆模型，而是要回答：FBBP 的 RAG 是否真的比 Direct 好，不同模型、不同 topK、不同题型上的收益是否稳定。

核心结论是：

> 在固定 `120` 题主表上，8 个模型全部出现 `RAG > Direct` 的回答相关性提升；在分类表里，`doc/design` 与 `schema_tables` 的平均提升最明显；在 `topK` 表里，`32 / 64 / 128` 存在明显质量-效率权衡；在 clean 稳定性表里，`MiniMax-M2.7` 质量更高，`DeepSeek-V3.2` 延迟更低。

## 评测目标

这个 benchmark 不是为了单纯比“谁更会聊天”，而是为了评估：

- 检索命中与重排对齐
- 回答相关性与证据一致性
- 跨表问答与结构化字段命中
- 真实 API 条件下的稳定性、时延与失败率

可用于论文的话术：

> 我们设计了一个受控领域评测协议，用来比较 FBBP RAG 与 Direct 的系统增益，并分析不同模型、不同题型和不同检索候选池大小对结果的影响。

## 正式协议

### 主表

- 8 个模型
- `Direct / RAG`
- 固定 `120` 题
- 受控推理协议

### 分类表

- 按 `ragppi`、`doc/design`、`schema_tables` 切分

### 附加表

- 同样的 8 个模型
- provider-native / supplementary 模式
- 只做补充，不替代主表

### topK 表

- `candidate_top_k = 32 / 64 / 128`
- `256` 已废弃，不进正式口径

### 稳定性表

- `DeepSeek-V3.2` 与 `MiniMax-M2.7`
- `100` 轮
- `RAG-only`
- clean release，`failed_rows_total = 0`

## 结果摘要

### 主表

| 模型 | Direct | RAG | 提升 | 备注 |
|---|---:|---:|---:|---|
| MiniMax-M2.7 | 0.0056 | 0.9072 | +0.9016 | 主表提升最大 |
| MiniMax-M2 | 0.0139 | 0.7701 | +0.7562 | 提升显著 |
| DeepSeek-R1 | 0.0682 | 0.7326 | +0.6644 | 提升显著 |
| GLM-5 | 0.1391 | 0.7772 | +0.6381 | 提升显著 |
| Qwen3-235B-A22B-Instruct-2507 | 0.1795 | 0.7388 | +0.5593 | 提升显著 |
| Kimi-K2.5 | 0.1649 | 0.6350 | +0.4701 | 提升显著 |
| ERNIE-4.5-Turbo-128K | 0.4061 | 0.7309 | +0.3248 | 提升稳定 |
| DeepSeek-V3.2 | 0.6398 | 0.7660 | +0.1262 | 提升较小但仍为正 |

### 分类表

| 类别 | 平均提升 | 中位提升 |
|---|---:|---:|
| doc/design | 0.7033 | 0.7673 |
| schema_tables | 0.5884 | 0.6550 |
| ragppi | 0.4079 | 0.3726 |

### topK 表

| topK | 平均 RAG answer relevancy | 平均提升 | 平均 RAG 延迟 |
|---|---:|---:|---:|
| 32 | 0.7614 | +0.5593 | 45138.84 ms |
| 64 | 0.7480 | +0.5459 | 77357.43 ms |
| 128 | 0.7146 | +0.4856 | 143530.31 ms |

结论：

> 候选池增大不是单调增益，质量和延迟要一起看。

### 稳定性表

| 模型 | answer_relevancy | 95% CI | 平均延迟 |
|---|---:|---:|---:|
| DeepSeek-V3.2 | 0.7624 | [0.7532, 0.7716] | 11009.04 ms |
| MiniMax-M2.7 | 0.8220 | [0.8085, 0.8356] | 24610.60 ms |

结论：

> MiniMax-M2.7 质量更高，DeepSeek-V3.2 更快。

## 结果边界

这条线要特别强调边界：

- 主表和稳定性表是正式口径，已经冻结。
- `reports/latest/*`、24 题 smoke、repair trace、quota repair trace 都不是正式结论。
- `topK=256` 不再属于正式协议。
- 附加表是补充分析，不替代主表。

## 方法

这个 benchmark 的方法可以概括成：

> 固定题集 + 受控推理协议 + 多模型对照 + Direct/RAG 对照 + topK 消融 + 类别切分 + clean 稳定性。

它的算法意义在于：

1. 把系统增益和模型生成能力分开。
2. 把结构化题型和开放问答分开。
3. 把检索候选池大小作为独立变量来分析。
4. 把稳定性和失败率作为正式指标，而不是只看单次均值。

## 论文可直接使用的方法段

我们构造了一个受控的 FBBP 领域评测协议，用于比较 RAG 与 Direct 在固定 120 题上的系统增益。主表采用 8 个模型、统一回答协议和统一推理约束，避免将 provider 的原生 thinking 行为误判为系统改进。与此同时，我们按题型构建分类表，对 `ragppi`、`doc/design` 和 `schema_tables` 三类问题分别分析，并通过 `candidate_top_k = 32 / 64 / 128` 的消融评估检索候选池大小对质量与延迟的影响。最后，我们使用 `DeepSeek-V3.2` 与 `MiniMax-M2.7` 做 100 轮 RAG-only clean 稳定性评测，以验证结论是否受单次 provider 抖动影响。

## 论文可直接使用的结果段

在固定 120 题主表上，8 个模型均表现出 `RAG > Direct` 的回答相关性提升，其中 `MiniMax-M2.7` 的提升最大（`0.0056 -> 0.9072`，`+0.9016`）。分类表进一步显示，RAG 对 `doc/design` 与 `schema_tables` 的提升最明显。`topK` 消融表表明，检索候选池扩大并不带来单调提升，且延迟显著增加。稳定性表显示，`MiniMax-M2.7` 在质量上领先，而 `DeepSeek-V3.2` 在延迟上更具优势，说明该系统存在清晰的质量-效率权衡。

## 简历表述

- 设计并冻结了 FBBP 领域 RAG 评测协议，比较 8 个模型在 Direct / RAG、分类题型、topK 消融和 clean 稳定性条件下的系统表现。
- 构建固定 120 题主表、分类表、附加表、topK 表和 100 轮 RAG-only 稳定性表，系统分析回答相关性、类别收益和延迟权衡。
- 发现 RAG 在 `doc/design` 与 `schema_tables` 上收益最明显，同时验证了候选池大小与延迟之间的显著权衡。

## 面试解释

短版：

> 我做的是一个正式的领域评测协议，不只是跑模型。它把 8 个模型、Direct/RAG、固定 120 题、分类题型、topK 消融和 clean 稳定性全部拆开，最后能清楚看到 RAG 在哪些题型上最有效、候选池多大最划算、哪些模型质量和效率之间怎么权衡。

## 证据文件

| 文件 | 作用 |
|---|---|
| `FINAL_RESULT_SUMMARY.md` | 官方总摘要 |
| `FINAL_RELEASE_GATE.md` | 发布门槛 |
| `reports/benchmark_final_summary_20260502/benchmark_results_final_summary_cn.md` | 中文正式结果 |
| `reports/benchmark_final_summary_20260502/benchmark_master_overview.csv` | 主表总览 |
| `reports/benchmark_final_summary_20260502/category_overall_uplift_summary.csv` | 分类表总览 |
| `reports/benchmark_final_summary_20260502/topk_overall_summary.csv` | topK 消融总览 |
| `reports/benchmark_final_summary_20260502/stability_final_summary.csv` | 稳定性表总览 |

