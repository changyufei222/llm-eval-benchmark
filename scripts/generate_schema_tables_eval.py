from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipelines.schema_benchmark_dataset import write_schema_benchmark_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 100-question benchmark dataset from schema_tables.")
    parser.add_argument(
        "--schema-dir",
        default=os.environ.get("SCHEMA_DIR", r"<local_path_removed>"),
        help="Normalized schema_tables directory.",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", r"<local_path_removed>"),
        help="Target JSONL dataset path.",
    )
    args = parser.parse_args()

    summary = write_schema_benchmark_dataset(Path(args.schema_dir), Path(args.output_path))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
