from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv

from metrics.metrics import frame_to_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_FILE = REPO_ROOT / "configs" / "paratera_text_models.txt"


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "model"


def parse_models_text(text: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        models.append(line)
        seen.add(line)
    return models


def load_models_file(path: Path) -> list[str]:
    return parse_models_text(path.read_text(encoding="utf-8"))


def resolve_models(explicit_models: Sequence[str], models_file: Path | None) -> list[str]:
    if explicit_models:
        return parse_models_text("\n".join(explicit_models))
    if models_file and models_file.exists():
        return load_models_file(models_file)
    raise FileNotFoundError(f"Model list file not found: {models_file}")


def apply_base_url_overrides(env: dict[str, str], base_url: str) -> None:
    normalized = base_url.rstrip("/")
    env["BASE_URL"] = normalized
    env["OPENAI_BASE_URL"] = normalized
    env["OPENAI_API_BASE"] = normalized


def _extract_summary_fields(summary: dict[str, Any]) -> dict[str, Any]:
    rag = summary.get("rag") or {}
    direct = summary.get("direct") or {}
    return {
        "eval_mode": summary.get("eval_mode"),
        "warning": summary.get("warning"),
        "rag_answer_relevancy": rag.get("answer_relevancy"),
        "rag_faithfulness": rag.get("faithfulness"),
        "rag_context_precision": rag.get("context_precision"),
        "rag_latency_ms": rag.get("latency_ms"),
        "rag_estimated_cost_usd": rag.get("estimated_cost_usd"),
        "direct_answer_relevancy": direct.get("answer_relevancy"),
        "direct_latency_ms": direct.get("latency_ms"),
        "direct_estimated_cost_usd": direct.get("estimated_cost_usd"),
    }


def _run_single_model(
    model: str,
    data_path: Path,
    output_root: Path,
    top_k: int,
    eval_mode: str,
    fail_on_fallback: bool,
    base_url: str | None,
    skip_existing: bool,
    stop_on_error: bool,
) -> dict[str, Any]:
    slug = slugify_model_name(model)
    model_output_dir = output_root / slug
    model_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = model_output_dir / "summary.json"
    if skip_existing and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "model": model,
            "slug": slug,
            "status": "cached",
            "returncode": 0,
            "output_dir": str(model_output_dir),
        }
        row.update(_extract_summary_fields(summary))
        return row

    env = os.environ.copy()
    env["LLM_MODEL"] = model
    if base_url:
        apply_base_url_overrides(env, base_url)

    command = [
        sys.executable,
        "-m",
        "pipelines.compare",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(model_output_dir),
        "--model",
        model,
        "--top-k",
        str(top_k),
        "--eval-mode",
        eval_mode,
    ]
    if fail_on_fallback:
        command.append("--fail-on-fallback")

    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (model_output_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (model_output_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

    row = {
        "model": model,
        "slug": slug,
        "status": "ok" if proc.returncode == 0 and summary_path.exists() else "failed",
        "returncode": proc.returncode,
        "output_dir": str(model_output_dir),
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row.update(_extract_summary_fields(summary))
    else:
        row["warning"] = "summary.json not produced"
    if proc.returncode != 0 and stop_on_error:
        raise RuntimeError(f"Model sweep failed for {model} with exit code {proc.returncode}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.model_sweep")
    parser.add_argument("--data-path", default="data/fbtp_eval.jsonl")
    parser.add_argument("--output-dir", default="reports/model_sweep_latest")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-mode", choices=["auto", "ragas", "local"], default="auto")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    models_file = Path(args.models_file) if args.models_file else None
    if models_file and not models_file.is_absolute():
        models_file = (REPO_ROOT / models_file).resolve()

    models = resolve_models(args.models, models_file)
    if not models:
        raise ValueError("No models configured for sweep.")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model in models:
        row = _run_single_model(
            model=model,
            data_path=data_path,
            output_root=output_root,
            top_k=args.top_k,
            eval_mode=args.eval_mode,
            fail_on_fallback=args.fail_on_fallback,
            base_url=args.base_url,
            skip_existing=args.skip_existing,
            stop_on_error=args.stop_on_error,
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "sweep_summary.csv", index=False)
    (output_root / "sweep_summary.md").write_text(frame_to_markdown(frame), encoding="utf-8")
    (output_root / "sweep_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model sweep summary to {output_root}")


if __name__ == "__main__":
    main()
