from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from metrics.formal_benchmark import combine_main_category_breakdowns
from metrics.metrics import frame_to_markdown


def write_category_summary(main_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = combine_main_category_breakdowns(main_dir)
    frame.to_csv(out_dir / "category_summary.csv", index=False)
    (out_dir / "category_summary.md").write_text(frame_to_markdown(frame), encoding="utf-8")
    (out_dir / "summary.md").write_text("# Category Summary\n\n" + frame_to_markdown(frame), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Write category summary artifacts from a merged main benchmark report.")
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    output = write_category_summary(Path(args.main_dir), Path(args.out_dir))
    print(output)


if __name__ == "__main__":
    main()
