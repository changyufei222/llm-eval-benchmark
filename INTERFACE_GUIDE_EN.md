# LLM Evaluation Benchmark Interface Guide

[English](./INTERFACE_GUIDE_EN.md) | [中文](./INTERFACE_GUIDE_CN.md)

## Purpose

Independent benchmark evidence layer for direct-vs-RAG comparison.

## Intended Readers

Algorithm reviewers, benchmark reviewers, and interviewers checking whether the results are controlled and reproducible.

## How To Read This Repository

- Start from README.md, FINAL_RELEASE_GATE.md, and FINAL_RESULT_SUMMARY.md.
- data/fbtp_eval_fixed_120.jsonl is the fixed benchmark set.
- reports/benchmark_final_summary_20260502/ is the main result evidence.
- pipelines/, metrics/, and tests/ show the benchmark implementation and validation.

## Repository Boundary

This repository is an upload-ready public package. Local paths, runtime caches, logs, private raw data, model weights, and temporary working files were excluded before upload.

## Language Switch

Use the links at the top of this file to switch between the English and Chinese interface guides.