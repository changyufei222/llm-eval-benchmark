#!/usr/bin/env bash
set -euo pipefail

# promptfoo (requires node)
if command -v npx >/dev/null 2>&1; then
  npx promptfoo eval -c configs/promptfoo.yaml
else
  echo "npx not found, skipping promptfoo."
fi

python -m pipelines.rag_pipeline --data-path data/fbtp_eval.jsonl --output-dir reports/latest
python -m pipelines.compare --data-path data/fbtp_eval.jsonl --output-dir reports/latest
