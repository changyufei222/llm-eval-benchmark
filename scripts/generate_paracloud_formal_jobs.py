from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark_protocol import (
    APPENDIX_NATIVE_MODELS,
    CONTROLLED_MAIN_MODELS,
    FINAL_CONTEXT_TOP_K,
    STABILITY_ANCHOR_MODELS,
    TOPK_CANDIDATE_VALUES,
    benchmark_model_flags,
)
from pipelines.benchmark_schedule import build_sampling_groups


SBATCH_HEADER = """#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
#SBATCH -J {job_name}
#SBATCH -o {logs_dir}/{job_name}_%j.out
#SBATCH -e {logs_dir}/{job_name}_%j.err
set -euo pipefail
export PGDATA={pgdata}
export PGSOCKET={pgsocket}
export PGLOG={logs_dir}/{pglog_prefix}_${{SLURM_JOB_ID}}.log
source {shared_scripts_dir}/common_env.sh
trap 'pg_ctl -D "$PGDATA" -m fast stop >/dev/null 2>&1 || true' EXIT
bash {shared_scripts_dir}/prepare_ingest.sh
"""


def _benchmark_model_flags(models: Sequence[str]) -> str:
    return benchmark_model_flags(models)


def _main_job_body(
    jobs_dir: Path,
    shared_scripts_dir: Path,
    reports_root: Path,
    time_limit: str,
) -> str:
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name="ragas_main_fixed120",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix="postgres_main_fixed120",
        shared_scripts_dir=shared_scripts_dir,
    )
    return (
        header
        + f"""STAMP=$(date +%Y%m%d_%H%M%S)
OUT={reports_root}/main_results_ragas_fixed120_$STAMP
mkdir -p "$OUT"
cd "$BENCH"
python -u -m pipelines.compare \\
  --data-path data/fbtp_eval_fixed_120.jsonl \\
  --output-dir "$OUT" \\
  --eval-mode ragas \\
  --fail-on-fallback \\
  --rounds 1 \\
  --top-k {FINAL_CONTEXT_TOP_K} \\
{_benchmark_model_flags(CONTROLLED_MAIN_MODELS)}
"""
    )


def _category_job_body(
    jobs_dir: Path,
    shared_scripts_dir: Path,
    reports_root: Path,
    time_limit: str,
) -> str:
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name="ragas_category_summary",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix="postgres_category_summary",
        shared_scripts_dir=shared_scripts_dir,
    )
    main_glob = reports_root / "main_results_ragas_fixed120_*"
    return (
        header
        + f"""MAIN_DIR=$(ls -dt {main_glob} 2>/dev/null | head -n 1)
if [ -z "${{MAIN_DIR:-}}" ]; then
  echo "main_results_ragas_fixed120_* not found under {reports_root}" >&2
  exit 2
fi
OUT={reports_root}/category_summary_fixed120_real
mkdir -p "$OUT"
cd "$BENCH"
python - <<PY
from pathlib import Path
from metrics.formal_benchmark import combine_main_category_breakdowns
from metrics.metrics import frame_to_markdown

main_dir = Path("$MAIN_DIR")
out_dir = Path("$OUT")
df = combine_main_category_breakdowns(main_dir)
df.to_csv(out_dir / "category_summary.csv", index=False)
(out_dir / "category_summary.md").write_text(frame_to_markdown(df), encoding="utf-8")
(out_dir / "summary.md").write_text("# Category Summary\\n\\n" + frame_to_markdown(df), encoding="utf-8")
PY
"""
    )


def _appendix_job_body(
    jobs_dir: Path,
    shared_scripts_dir: Path,
    reports_root: Path,
    time_limit: str,
) -> str:
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name="ragas_appendix_native",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix="postgres_appendix_native",
        shared_scripts_dir=shared_scripts_dir,
    )
    return (
        header
        + f"""STAMP=$(date +%Y%m%d_%H%M%S)
OUT={reports_root}/appendix_native_thinking_fixed120_$STAMP
mkdir -p "$OUT"
cd "$BENCH"
export BENCHMARK_DIRECT_THINKING_MODE=enabled
export RAGKB_OPENAI_THINKING_MODE=enabled
export BENCHMARK_DIRECT_MAX_TOKENS=${{BENCHMARK_DIRECT_MAX_TOKENS:-1024}}
python -u -m pipelines.compare \\
  --data-path data/fbtp_eval_fixed_120.jsonl \\
  --output-dir "$OUT" \\
  --eval-mode ragas \\
  --fail-on-fallback \\
  --rounds 1 \\
  --top-k {FINAL_CONTEXT_TOP_K} \\
{_benchmark_model_flags(APPENDIX_NATIVE_MODELS)}
"""
    )


def _sampling_job_body(
    jobs_dir: Path,
    shared_scripts_dir: Path,
    reports_root: Path,
    group: dict[str, int | str],
    base_seed: int,
    time_limit: str,
) -> str:
    seed = base_seed + int(group["round_start"]) - 1
    out_dir = reports_root / "sampling" / str(group["group_name"])
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name=f"ragas_{group['group_name']}",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix=f"postgres_{group['group_name']}",
        shared_scripts_dir=shared_scripts_dir,
    )
    return (
        header
        + f"""OUT={out_dir}
mkdir -p "$OUT"
cd "$BENCH"
python -u -m pipelines.compare \\
  --data-path data/fbtp_eval_fixed_120.jsonl \\
  --output-dir "$OUT" \\
  --eval-mode ragas \\
  --fail-on-fallback \\
  --rounds {group['round_count']} \\
  --sample-size 50 \\
  --with-replacement \\
  --seed {seed} \\
  --top-k {FINAL_CONTEXT_TOP_K} \\
{_benchmark_model_flags(STABILITY_ANCHOR_MODELS)}
"""
    )


def _topk_job_body(
    jobs_dir: Path,
    shared_scripts_dir: Path,
    reports_root: Path,
    candidate_top_k: int,
    time_limit: str,
) -> str:
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name=f"ragas_topk_{candidate_top_k}",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix=f"postgres_topk_{candidate_top_k}",
        shared_scripts_dir=shared_scripts_dir,
    )
    out_dir = reports_root / "topk_candidate_ablation_fixed120" / f"candidate_topk_{candidate_top_k:03d}"
    return (
        header
        + f"""OUT={out_dir}
mkdir -p "$OUT"
cd "$BENCH"
python -u -m pipelines.compare \\
  --data-path data/fbtp_eval_fixed_120.jsonl \\
  --output-dir "$OUT" \\
  --eval-mode ragas \\
  --fail-on-fallback \\
  --rounds 1 \\
  --top-k {FINAL_CONTEXT_TOP_K} \\
  --candidate-top-k {candidate_top_k} \\
{_benchmark_model_flags(CONTROLLED_MAIN_MODELS)}
"""
    )


def _aggregate_job_body(jobs_dir: Path, shared_scripts_dir: Path, reports_root: Path, time_limit: str) -> str:
    header = SBATCH_HEADER.format(
        time_limit=time_limit,
        job_name="ragas_aggregate",
        logs_dir=jobs_dir / "logs",
        pgdata="/data/run01/scv7sd2/postgres/benchmark_main",
        pgsocket="/data/run01/scv7sd2/postgres/benchmark_socket",
        pglog_prefix="postgres_aggregate",
        shared_scripts_dir=shared_scripts_dir,
    )
    return (
        header
        + f"""OUT={reports_root / 'final'}
mkdir -p "$OUT"
cd "$BENCH"
python scripts/aggregate_formal_benchmark.py \\
  --main-glob '{reports_root / 'main_results_ragas_fixed120_*'}' \\
  --sampling-root '{reports_root / 'sampling'}' \\
  --output-dir "$OUT"
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formal benchmark sbatch files for the Paracloud N26 environment.")
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--shared-scripts-dir", type=Path, default=Path("/data/run01/scv7sd2/jobs/scripts"))
    parser.add_argument("--total-rounds", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-parallel-groups", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--sampling-time-limit", type=str, default="18:00:00")
    parser.add_argument("--aggregate-time-limit", type=str, default="02:00:00")
    parser.add_argument("--main-time-limit", type=str, default="18:00:00")
    parser.add_argument("--category-time-limit", type=str, default="02:00:00")
    parser.add_argument("--appendix-time-limit", type=str, default="18:00:00")
    parser.add_argument("--topk-time-limit", type=str, default="18:00:00")
    args = parser.parse_args()

    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    (args.jobs_dir / "logs").mkdir(parents=True, exist_ok=True)
    args.reports_root.mkdir(parents=True, exist_ok=True)
    (args.reports_root / "sampling").mkdir(parents=True, exist_ok=True)

    groups = build_sampling_groups(
        total_rounds=args.total_rounds,
        group_size=args.group_size,
        max_parallel_groups=args.max_parallel_groups,
    )

    manifest = {
        "reports_root": str(args.reports_root),
        "base_seed": args.base_seed,
        "controlled_main_models": list(CONTROLLED_MAIN_MODELS),
        "stability_anchor_models": list(STABILITY_ANCHOR_MODELS),
        "appendix_native_models": list(APPENDIX_NATIVE_MODELS),
        "topk_candidate_values": list(TOPK_CANDIDATE_VALUES),
        "final_context_top_k": FINAL_CONTEXT_TOP_K,
        "groups": groups,
    }
    (args.jobs_dir / "formal_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (args.jobs_dir / "run_main_formal_benchmark.sbatch").write_text(
        _main_job_body(
            jobs_dir=args.jobs_dir,
            shared_scripts_dir=args.shared_scripts_dir,
            reports_root=args.reports_root,
            time_limit=args.main_time_limit,
        ),
        encoding="utf-8",
    )
    (args.jobs_dir / "run_category_summary_from_main.sbatch").write_text(
        _category_job_body(
            jobs_dir=args.jobs_dir,
            shared_scripts_dir=args.shared_scripts_dir,
            reports_root=args.reports_root,
            time_limit=args.category_time_limit,
        ),
        encoding="utf-8",
    )
    (args.jobs_dir / "run_appendix_native_thinking.sbatch").write_text(
        _appendix_job_body(
            jobs_dir=args.jobs_dir,
            shared_scripts_dir=args.shared_scripts_dir,
            reports_root=args.reports_root,
            time_limit=args.appendix_time_limit,
        ),
        encoding="utf-8",
    )

    for group in groups:
        script_path = args.jobs_dir / f"run_sampling_{group['group_name']}.sbatch"
        script_path.write_text(
            _sampling_job_body(
                jobs_dir=args.jobs_dir,
                shared_scripts_dir=args.shared_scripts_dir,
                reports_root=args.reports_root,
                group=group,
                base_seed=args.base_seed,
                time_limit=args.sampling_time_limit,
            ),
            encoding="utf-8",
        )

    for candidate_top_k in TOPK_CANDIDATE_VALUES:
        (args.jobs_dir / f"run_topk_candidate_{candidate_top_k:03d}.sbatch").write_text(
            _topk_job_body(
                jobs_dir=args.jobs_dir,
                shared_scripts_dir=args.shared_scripts_dir,
                reports_root=args.reports_root,
                candidate_top_k=candidate_top_k,
                time_limit=args.topk_time_limit,
            ),
            encoding="utf-8",
        )

    (args.jobs_dir / "run_aggregate_formal_benchmark.sbatch").write_text(
        _aggregate_job_body(
            jobs_dir=args.jobs_dir,
            shared_scripts_dir=args.shared_scripts_dir,
            reports_root=args.reports_root,
            time_limit=args.aggregate_time_limit,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
