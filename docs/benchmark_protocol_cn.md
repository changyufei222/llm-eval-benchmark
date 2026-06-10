# Benchmark 规则说明（中文）

## 1. 目的

本 benchmark 的核心目标不是单纯比较“哪个模型最会推理”，而是稳定评估本项目 RAG 系统相对于 Direct 基线的真实增益，包括：

- 检索命中与重排对齐效果
- 回答相关性与证据一致性
- 跨表问答与结构化字段命中能力
- 真实 API 条件下的稳定性、时延与失败率

因此，主结果表、稳定性表、分类表采用统一的受控推理协议，尽量减少不同模型默认 thinking 策略造成的额外干扰。附录表与 `topK` 消融表单独运行，用于回答“原生推理模式是否改变结论”以及“检索候选池大小是否影响系统增益”这两类独立问题。

## 2. 主结果表

主结果表定义为：

- 8 个模型
- 2 种方法：`Direct / RAG`
- 固定 `120` 题全集
- 指标以 `RAGAS` 为主，并保留本地结构化指标用于辅助分析

模型列表：

- `DeepSeek-V3.2`
- `GLM-5`
- `Kimi-K2.5`
- `DeepSeek-R1`
- `Qwen3-235B-A22B-Instruct-2507`
- `MiniMax-M2`
- `MiniMax-M2.7`
- `ERNIE-4.5-Turbo-128K`

## 3. 主表推理协议

主表采用以下受控推理策略：

- `DeepSeek-V3.2`：关闭 thinking
- `GLM-5`：关闭 thinking
- `Kimi-K2.5`：关闭 thinking
- `DeepSeek-R1`：保留原生推理行为，不强制关闭 thinking
- `Qwen3-235B-A22B-Instruct-2507`：采用受控非 thinking 协议
- `MiniMax-M2`：采用受控非 thinking 协议
- `MiniMax-M2.7`：采用受控非 thinking 协议
- `ERNIE-4.5-Turbo-128K`：采用受控非 thinking 协议

说明：

- `DeepSeek-V3.2 / GLM-5 / Kimi-K2.5` 在 Direct 与 RAG 两条回答链中都使用相同的 thinking 关闭策略。
- `DeepSeek-R1` 保留原生 reasoning，是因为该模型在兼容接口下对强行关闭 thinking 或极小输出预算较敏感，容易出现可见答案异常或失真，因此在主表中保留其原生行为。
- 对 `Qwen3-235B-A22B-Instruct-2507 / MiniMax-M2 / MiniMax-M2.7 / ERNIE-4.5-Turbo-128K`，若 provider 支持显式 thinking 开关，则关闭 thinking；若不支持显式开关，则使用默认普通回答模式，不额外请求展示思维链。
- 所有主表模型在 Direct 与 RAG 两条链中必须使用完全一致的回答协议，避免把模型模式差异误判为系统增益。

## 4. 稳定性表

本次正式晋升的稳定性表采用 selected-RAG clean release，不再把“更大但仍带修复痕迹的多模型重复实验”直接当成唯一官方口径。

定义为：

- 模型：`DeepSeek-V3.2 / MiniMax-M2.7`
- 基础题集：固定 `120` 题总体
- 每轮：有放回抽样 `50` 题
- 共 `100` 轮
- 路径：仅统计 `RAG`
- 输出统计：
  - `mean`
  - `p50`
  - `p95`
  - `95% CI`
  - `failed_rows_total`
  - `full_success_rounds`

稳定性表用于衡量：

- 在 clean 条件下，两种代表性模型的质量-效率权衡是否稳定
- 结果是否已经摆脱 provider 限流、额度和中间 repair 痕迹的污染
- 最终可交付版里哪些模型应被解释为“质量优先”与“效率优先”

说明：

- 更早的 `4` anchor model 稳定性设计和中间运行记录保留为工程历史，可用于内部追溯，但不再作为本次正式 release 的唯一官方结论。
- 正式晋升后的稳定性结论，应以 `reports/benchmark_final_summary_20260502/stability_final_summary.csv` 及其 clean 校验文件为准。

## 5. 分类结果表

分类结果表同样沿用主表协议，模型集合与主表完全一致，在固定 `120` 题全集上按题型拆分统计，当前主要包含：

- `ragppi`
- `doc/design`
- `schema_tables`

该表用于分析不同类型问题上 RAG 是否存在结构性优势或短板。

## 6. 附录 / 补充表

为避免把模型 provider-native 行为与主 benchmark 的系统增益混在一起，补充表单独运行，不并入主表、稳定性表和分类表。

附录表定义为：

- 模型：与主表完全相同的 8 个模型
- 模式：`provider-native / native-thinking`
- 用途：作为补充分析，展示同一批模型在 provider 原生回答协议下的表现

该表回答的问题是：

- 如果允许模型使用 provider 原生回答协议，它们的 Direct / RAG 行为会如何变化？
- 主表中的受控协议与 provider-native 模式之间差距有多大？

说明：

- 若模型支持显式 thinking 或 reasoning 模式，附录表中不主动压制该能力。
- 若模型本身没有显式 thinking 开关，则使用 provider 默认普通回答协议；这类模型仍保留在附录表中，以保证附录表与主表的模型阵容完全一致。
- 因此，附录表中的 “native-thinking” 是一个统称，实际含义是“使用 provider-native 默认回答协议”，而不是要求所有模型都显式输出思维链。

## 7. `topK` 消融表

`topK` 消融表单独运行，不并入主表、分类表、附录表和稳定性表。

定义为：

- 模型：与主表完全相同的 8 个模型
- 题集：固定 `120` 题全集
- 方法：只对 `RAG` 路径做检索参数消融；`Direct` 基线复用主表结果
- 变量：检索候选池 `topK ∈ {32, 64, 128}`
- 固定项：
  - 生成侧最终上下文预算保持不变
  - prompt compression 策略保持不变
  - 其余回答协议与主表一致

该表用于回答：

- 增大检索候选池是否会带来稳定的质量提升？
- 候选池变大后，延迟、失败率与 token 开销是否值得？
- `RAG(topK) - Direct(main)` 的提升是否对检索候选规模敏感？

输出建议至少包含：

- `RAGAS` 主指标
- 本地结构化命中指标
- latency
- failure rate
- `RAG(topK) - Direct(main)` 平均提升

## 8. 结果解释原则

解释 benchmark 结果时遵循以下原则：

- 主结论以主结果表、分类表和正式晋升后的 clean 稳定性表为准
- 附录表只作为补充，不替代主结果表
- `topK` 消融表只解释检索配置敏感性，不替代主表排序
- 若某个模型在主表中表现不如附录，不应简单解释为“模型更差”，而应结合受控协议与原生策略差异分析
- 若某个模型在更高候选池下优于 `topK=32`，应同时检查延迟、失败率与 token 代价，不能只看单一质量分数
- 若某个指标出现 `RAG < Direct`，应结合证据命中、表命中、ID 命中、时延与失败率综合判断，避免只看单一分数

## 9. 当前执行口径

自 `2026-05-02` 起，正式对外口径按如下规则收束：

- 主结果表：`8` 模型，受控协议
- 稳定性表：`DeepSeek-V3.2 / MiniMax-M2.7`，`100` 轮，`RAG-only` clean release
- 分类表：与主表相同的 `8` 模型，受控协议
- 附录表：与主表相同的 `8` 模型，provider-native / native-thinking
- `topK` 消融表：与主表相同的 `8` 模型，单独扫描 `32 / 64 / 128`

过渡说明：

- 服务器上更早的 `4` 模型 anchor stability 记录保留为工程历史，但不再与正式 release 口径混用。
- 先前的中间重跑、repair、quota 恢复和 backup 目录只保留为审计证据，不进入正式对外结论。
- `Baichuan-M3` 已从主表与附录模型阵容中移除，替换为 `MiniMax-M2.7`。原因不是模型质量结论，而是当前 provider 的 OpenAI-compatible thinking 控制接口对 `Baichuan-M3` 存在 `budget_tokens` 参数兼容问题，会污染受控协议下的主表结果。
- 对已经在远端完成的 `Qwen3-235B-A22B-Instruct-2507 / MiniMax-M2` 主表结果，增量运行采用“复用已完成产物 + 只运行新增模型”的方式继续，不重复消耗同一批已完成结果。
- `topK=256` 不再属于正式 benchmark 协议。原因不是方法学失效，而是该档位在真实 API 条件下成本与时延显著膨胀，且对论文主结论的边际增益有限。后续论文、答辩与正式对外结果统一只采用 `32 / 64 / 128` 三档候选池。

这份文档是当前 benchmark 的唯一口径说明，后续论文、答辩与结果解读均应与本文件保持一致。
