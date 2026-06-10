from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from metrics.metrics import compute_metrics, metrics_to_markdown
from pipelines._bootstrap_ragkb import ensure_ragkb_on_path

ensure_ragkb_on_path()

from ragkb.openai_compat import adjusted_completion_max_tokens, openai_thinking_mode


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_base_url() -> str | None:
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "BASE_URL"):
        value = os.getenv(key)
        if value:
            return value.rstrip("/")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipelines.direct_pipeline")
    parser.add_argument("--data-path", default="data/fbtp_eval.jsonl")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv()
    kwargs = {}
    base_url = _resolve_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    model = args.model or os.getenv("LLM_MODEL", "gpt-5.4")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    rows = _load_jsonl(data_path)
    results = []

    for row in rows:
        question = row.get("question", "")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Answer the question concisel<local_path_removed>",
                }
            ],
            max_tokens=adjusted_completion_max_tokens(model, 128),
            **(
                {"extra_body": {"thinking": {"type": thinking_mode}}}
                if (thinking_mode := openai_thinking_mode(model))
                else {}
            ),
        )
        answer = (response.choices[0].message.content or "").strip()
        results.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": row.get("ground_truth", ""),
                "answer_chars": len(answer),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "direct_answers.jsonl").write_text(
        "\n".join(json.dumps(result, ensure_ascii=False) for result in results),
        encoding="utf-8",
    )

    summary = compute_metrics(
        results,
        method="Direct",
        extra={"samples": len(results), "model": model, "data_path": str(data_path)},
    )
    (out_dir / "direct_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "direct_summary.md").write_text(
        metrics_to_markdown("# Direct Answering Summary", summary),
        encoding="utf-8",
    )
    print("Saved direct answers to", out_dir)


if __name__ == "__main__":
    main()
