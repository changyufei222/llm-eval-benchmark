from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.metrics import frame_to_markdown
from metrics.plotting import write_metric_plot
from pipelines.compare import _build_leaderboard


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _collect_model_specs(report_dir: Path) -> list[dict[str, str]]:
    config = _load_json(report_dir / "experiment_config.json")
    return list(config.get("benchmark_models") or [])


def _collect_summary_payload(report_dir: Path) -> dict[str, Any]:
    return _load_json(report_dir / "summary.json")


def merge_multimodel_reports(input_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if not input_dirs:
        raise ValueError("input_dirs must not be empty")

    ordered_specs: list[dict[str, str]] = []
    seen_slugs: set[str] = set()
    model_summary_frames: list[pd.DataFrame] = []
    per_round_frames: list[pd.DataFrame] = []
    sampling_round_frames: list[pd.DataFrame] = []
    base_summary = _collect_summary_payload(input_dirs[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = output_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    for report_dir in input_dirs:
        for spec in _collect_model_specs(report_dir):
            slug = spec["slug"]
            if slug in seen_slugs:
                raise ValueError(f"Duplicate model slug during merge: {slug}")
            seen_slugs.add(slug)
            ordered_specs.append(spec)
            _copytree_replace(report_dir / "models" / slug, models_root / slug)

        frame = _read_optional_csv(report_dir / "model_summary.csv")
        if not frame.empty:
            model_summary_frames.append(frame)
        frame = _read_optional_csv(report_dir / "per_round_results.csv")
        if not frame.empty:
            per_round_frames.append(frame)
        frame = _read_optional_csv(report_dir / "sampling_rounds.csv")
        if not frame.empty:
            sampling_round_frames.append(frame)

    model_summary = pd.concat(model_summary_frames, ignore_index=True) if model_summary_frames else pd.DataFrame()
    per_round = pd.concat(per_round_frames, ignore_index=True) if per_round_frames else pd.DataFrame()
    sampling_rounds = pd.concat(sampling_round_frames, ignore_index=True) if sampling_round_frames else pd.DataFrame()
    leaderboard = _build_leaderboard(model_summary)

    experiment_config = {
        "data_path": base_summary.get("data_path"),
        "population_size": base_summary.get("population_size"),
        "rounds": base_summary.get("rounds"),
        "sample_size": base_summary.get("sample_size"),
        "with_replacement": base_summary.get("with_replacement"),
        "seed": base_summary.get("seed"),
        "benchmark_models": ordered_specs,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    (output_dir / "model_summary.md").write_text(frame_to_markdown(model_summary), encoding="utf-8")
    model_summary.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.md").write_text(frame_to_markdown(model_summary), encoding="utf-8")
    if not model_summary.empty:
        write_metric_plot(model_summary, output_dir / "comparison_summary.png", title="Merged RAG vs Direct Comparison")
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    (output_dir / "leaderboard.md").write_text(frame_to_markdown(leaderboard), encoding="utf-8")
    per_round.to_csv(output_dir / "per_round_results.csv", index=False)
    (output_dir / "per_round_results.md").write_text(frame_to_markdown(per_round), encoding="utf-8")
    sampling_rounds.to_csv(output_dir / "sampling_rounds.csv", index=False)
    (output_dir / "sampling_rounds.md").write_text(frame_to_markdown(sampling_rounds), encoding="utf-8")

    summary = {
        "data_path": base_summary.get("data_path"),
        "population_size": base_summary.get("population_size"),
        "model_count": len(ordered_specs),
        "rounds": base_summary.get("rounds"),
        "sample_size": base_summary.get("sample_size"),
        "with_replacement": base_summary.get("with_replacement"),
        "seed": base_summary.get("seed"),
        "experiment_config_path": str(output_dir / "experiment_config.json"),
        "per_round_results_path": str(output_dir / "per_round_results.csv"),
        "model_summary_path": str(output_dir / "model_summary.csv"),
        "leaderboard_path": str(output_dir / "leaderboard.csv"),
        "models": [
            {
                "label": spec["label"],
                "model": spec["model"],
                "slug": spec["slug"],
                "summary_path": str(output_dir / "models" / spec["slug"] / "summary.json"),
            }
            for spec in ordered_specs
        ],
        "merged_from": [str(path) for path in input_dirs],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# Merged Multi-Model Summary\n\n"
        + frame_to_markdown(leaderboard)
        + "\n\n## Model Summary\n\n"
        + frame_to_markdown(model_summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple multimodel benchmark report directories into one combined report.")
    parser.add_argument("--input-dir", action="append", required=True, help="Input multimodel report directory. Repeat for multiple directories.")
    parser.add_argument("--output-dir", required=True, help="Output merged report directory.")
    args = parser.parse_args()

    merge_multimodel_reports(
        [Path(path) for path in args.input_dir],
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
