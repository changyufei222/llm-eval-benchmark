from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def filter_multimodel_report(
    input_dir: Path,
    output_dir: Path,
    *,
    model_slugs: Iterable[str],
) -> dict[str, Any]:
    config = _load_json(input_dir / "experiment_config.json")
    selected = set(model_slugs)
    specs = [spec for spec in list(config.get("benchmark_models") or []) if spec.get("slug") in selected]
    if not specs:
        raise ValueError("No requested model slugs were found in experiment_config.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = output_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    model_summary_frames: list[pd.DataFrame] = []
    per_round_frames: list[pd.DataFrame] = []
    sampling_round_frames: list[pd.DataFrame] = []

    copied_specs: list[dict[str, Any]] = []
    for spec in specs:
        slug = str(spec["slug"])
        src_model_dir = input_dir / "models" / slug
        if not src_model_dir.exists():
            continue
        _copytree_replace(src_model_dir, models_root / slug)
        copied_specs.append(spec)
        frame = _read_optional_csv(src_model_dir / "model_summary.csv")
        if not frame.empty:
            model_summary_frames.append(frame)
        frame = _read_optional_csv(src_model_dir / "per_round_results.csv")
        if not frame.empty:
            per_round_frames.append(frame)
        frame = _read_optional_csv(src_model_dir / "sampling_rounds.csv")
        if not frame.empty:
            sampling_round_frames.append(frame)

    if not copied_specs:
        raise ValueError("None of the requested model directories exist in the input report")

    filtered_config = dict(config)
    filtered_config["benchmark_models"] = copied_specs
    (output_dir / "experiment_config.json").write_text(
        json.dumps(filtered_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_summary = pd.concat(model_summary_frames, ignore_index=True) if model_summary_frames else pd.DataFrame()
    per_round = pd.concat(per_round_frames, ignore_index=True) if per_round_frames else pd.DataFrame()
    sampling_rounds = (
        pd.concat(sampling_round_frames, ignore_index=True) if sampling_round_frames else pd.DataFrame()
    )
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    per_round.to_csv(output_dir / "per_round_results.csv", index=False)
    sampling_rounds.to_csv(output_dir / "sampling_rounds.csv", index=False)

    summary = {
        "source_report_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_count": len(copied_specs),
        "benchmark_models": copied_specs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a multi-model report down to a subset of completed model slugs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-slug", action="append", required=True)
    args = parser.parse_args()

    filter_multimodel_report(
        Path(args.input_dir),
        Path(args.output_dir),
        model_slugs=args.model_slug,
    )


if __name__ == "__main__":
    main()
