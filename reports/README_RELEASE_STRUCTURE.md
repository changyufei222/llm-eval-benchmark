# Reports Release Structure

This file explains which folders under `reports/` are part of the promoted benchmark release and which folders remain only for smoke runs, engineering traces, or historical auditability.

## Official Release Surface

Use these paths for external sharing, portfolio narratives, final benchmark writeups, and teacher/interview summaries:

- `benchmark_final_summary_20260502/`
- `../FINAL_RESULT_SUMMARY.md`
- `../FINAL_RELEASE_GATE.md`

Within `benchmark_final_summary_20260502/`, the canonical files are:

- `benchmark_results_final_summary_cn.md`
- `benchmark_results_final_summary.md`
- `benchmark_master_overview.csv`
- `main_table_ranked_summary.csv`
- `category_overall_uplift_summary.csv`
- `appendix_ranked_summary.csv`
- `topk_overall_summary.csv`
- `topk_best_k_by_model.csv`
- `stability_final_summary.csv`

## Remote Sync Results

`remote_sync/` keeps raw synced benchmark trees that were needed to reconstruct and verify final results. These directories are not all equal:

- Official promoted sources:
  - `remote_sync/main_results_ragas_fixed120_controlled8_merged_20260418_133150`
  - `remote_sync/stability_selected_rag_best_vs_worst_20260423_133411`
- Audit-only backups:
  - paths containing `preclean_backup`
  - paths containing `prefinal_sync_backup`
- Incomplete / non-promoted intermediate trees:
  - `remote_sync/main_results_ragas_fixed120_missing2_20260418_133150`

See `remote_sync/README_RELEASE_STATUS.md` for the exact boundary.

## Smoke And Reproducibility Runs

These folders are useful for local validation, end-to-end checks, or earlier smoke demonstrations, but they are not the official promoted benchmark conclusion:

- `latest/`
- `latest_smoke/`
- `latest_smoke_codex/`
- `latest_smoke_offline_flags/`
- `fixed_sampling_smoke_local/`
- `fixed_sampling_smoke_ragas/`
- `fixed_sampling_smoke_ragas_venv/`
- `fixed120_smoke_local/`
- `fixed120_smoke_local_v2/`
- `fixed120_smoke_ragas_v2/`
- `model_sweep_smoke/`
- `model_sweep_smoke_fixed/`
- `multimodel_smoke_20260411/`
- `multimodel_smoke_20260411_min/`
- `multimodel_smoke_20260411_min_v2/`
- `multimodel_smoke_ragas_20260411/`

## Historical Or Engineering-Only Experiment Trees

These directories remain for reproducibility and engineering traceability, but they do not define the current official benchmark narrative:

- `formal_ragas_4models_100rounds_50sample_20260411/`
- `formal_ragas_4models_100rounds_50sample_20260411_clean/`
- `full_ragas_nofallback_20260324/`
- `run_bge_gpt54/`

Important note:

- `formal_ragas_4models_100rounds_50sample_20260411_clean/` does not contain the final promoted aggregate files such as `summary.json`, `model_summary.csv`, `per_round_results.csv`, or `leaderboard.csv`, so it should not be cited as the official final stability release.

## Dashboard Surface

`control_plane_dashboard/` is an engineering support surface for eval routing and control-plane inspection. It is not a polished product UI and is not the benchmark release gate.

See `control_plane_dashboard/README_STATUS.md`.
