# LLM Eval Benchmark

[English](./README.md) | **中文**

这是一个 CLI 优先、可审计的 LLM 与 RAG 评估仓库，用于比较 Direct 与 RAG 回答、不同模型、检索参数和稳定性设置。正式结论与本地 smoke 输出严格分开，避免把诊断运行误当作发布结果。

## 正式结果范围

- 主表：固定 120 问题集，8 个模型，Direct/RAG 两种方法
- 稳定性：选定模型的 RAG-only 重复运行
- 分类结果：agppi、doc-design、schema_tables
- 附录：provider-native 模式
- 检索消融：候选池 32 / 64 / 128

## 快速导航

| 目标 | 入口 |
|---|---|
| 查看最终结论 | [FINAL_RESULT_SUMMARY.md](./FINAL_RESULT_SUMMARY.md) |
| 查看发布门控 | [FINAL_RELEASE_GATE.md](./FINAL_RELEASE_GATE.md) |
| 阅读中文协议 | [docs/benchmark_protocol_cn.md](./docs/benchmark_protocol_cn.md) |
| 获取固定数据集 | [data/fbtp_eval_fixed_120.jsonl](./data/fbtp_eval_fixed_120.jsonl) |
| 查看正式报告 | [eports/benchmark_final_summary_20260502/](./reports/benchmark_final_summary_20260502/) |

## 复现边界

- eports/latest/ 是 smoke/复现输出，不是正式结论入口。
- data/fbtp_eval_fixed_120.jsonl 是正式固定集。
- 外部模型 API 的可重复性受模型版本、供应商和推理参数影响。
- 运行时密钥必须通过环境变量提供，不应提交到仓库。

详细界面说明见 [INTERFACE_GUIDE_CN.md](./INTERFACE_GUIDE_CN.md)。
