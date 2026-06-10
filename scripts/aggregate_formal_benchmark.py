from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.formal_benchmark import (
    combine_main_category_breakdowns,
    summarize_benchmark_distribution,
    summarize_rag_direct_uplift,
)
from metrics.metrics import frame_to_markdown


def _resolve_latest_glob(pattern: str) -> Path:
    matches = sorted((Path(match) for match in glob.glob(pattern)), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No paths matched pattern: {pattern}")
    return matches[-1]


def _load_sampling_group_frames(sampling_root: Path) -> pd.DataFrame:
    group_dirs = sorted(
        [path for path in sampling_root.glob("group_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if not group_dirs:
        raise FileNotFoundError(f"No sampling group directories found under: {sampling_root}")

    frames: list[pd.DataFrame] = []
    round_offset = 0
    for group_index, group_dir in enumerate(group_dirs, start=1):
        csv_path = group_dir / "per_round_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing per_round_results.csv in {group_dir}")
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        frame["group_name"] = group_dir.name
        frame["group_index"] = group_index
        local_rounds = sorted(frame["round"].dropna().astype(int).unique().tolist()) if "round" in frame.columns else []
        round_map = {local_round: round_offset + offset for offset, local_round in enumerate(local_rounds, start=1)}
        if round_map:
            frame["global_round"] = frame["round"].map(round_map)
            round_offset += len(local_rounds)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_frame(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(frame_to_markdown(df), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fixed120 main results and split repeated-sampling groups.")
    parser.add_argument("--main-dir", type=Path, default=None, help="Exact main fixed120 result directory.")
    parser.add_argument("--main-glob", type=str, default=None, help="Glob used to resolve the latest main fixed120 result directory.")
    parser.add_argument("--sampling-root", type=Path, required=True, help="Root directory containing group_XX subdirectories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write final aggregated artifacts.")
    args = parser.parse_args()

    if args.main_dir is None and args.main_glob is None:
        raise SystemExit("One of --main-dir or --main-glob is required.")

    if args.main_dir is not None:
        main_dir = args.main_dir
    else:
        main_dir = _resolve_latest_glob(args.main_glob)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rounds = _load_sampling_group_frames(args.sampling_root)
    _write_frame(combined_rounds, output_dir / "sampling_per_round_results.csv", output_dir / "sampling_per_round_results.md")

    stability_summary = summarize_benchmark_distribution(
        combined_rounds,
        group_cols=["model_label", "model", "method"],
        exclude_numeric_cols={"round", "seed", "sample_size", "unique_questions", "rows", "group_index", "global_round"},
    )
    uplift_summary = summarize_rag_direct_uplift(combined_rounds)
    category_summary = combine_main_category_breakdowns(main_dir)

    _write_frame(stability_summary, output_dir / "stability_summary.csv", output_dir / "stability_summary.md")
    _write_frame(uplift_summary, output_dir / "rag_minus_direct_uplift.csv", output_dir / "rag_minus_direct_uplift.md")
    _write_frame(category_summary, output_dir / "category_summary.csv", output_dir / "category_summary.md")

    for artifact_name in ("model_summary.csv", "model_summary.md", "leaderboard.csv", "leaderboard.md", "summary.json", "summary.md"):
        source_path = main_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / f"main_fixed120_{artifact_name}")

    summary_md = "\n\n".join(
        [
            f"# Formal Benchmark Aggregate\n\n- main_dir: `{main_dir}`\n- sampling_root: `{args.sampling_root}`\n- output_dir: `{output_dir}`",
            "## Main Fixed120 Table\n\nSee `main_fixed120_model_summary.csv` / `main_fixed120_leaderboard.csv`.",
            "## Stability Summary\n\n" + frame_to_markdown(stability_summary),
            "## RAG - Direct Uplift\n\n" + frame_to_markdown(uplift_summary),
            "## Category Summary\n\n" + frame_to_markdown(category_summary),
        ]
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")


if __name__ == "__main__":
    main()
