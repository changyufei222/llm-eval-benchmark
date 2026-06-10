# LLM 评测基准 界面说明

[English](./INTERFACE_GUIDE_EN.md) | [中文](./INTERFACE_GUIDE_CN.md)

## 这个仓库是做什么的

用于 Direct vs RAG 对照的独立评测证据层。

## 谁应该先看这个说明

算法评审者、benchmark 评审者，以及关注结果是否受控和可复现的面试官。

## 仓库阅读顺序

- 先读 README.md、FINAL_RELEASE_GATE.md 和 FINAL_RESULT_SUMMARY.md。
- data/fbtp_eval_fixed_120.jsonl 是固定评测题集。
- reports/benchmark_final_summary_20260502/ 是主结果证据。
- pipelines/、metrics/ 和 tests/ 展示评测实现与验证。

## 上传边界

这个仓库是已经整理过的公开上传版本。上传前已经排除了本机路径、运行缓存、日志、原始私有数据、模型权重和临时工作文件。

## 中英文切换

本文件顶部提供 English / 中文链接，可在英文界面说明和中文界面说明之间切换。