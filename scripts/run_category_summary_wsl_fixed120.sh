#!/bin/bash
set -euo pipefail

cd <local_path_removed>

BASE_MAIN_OUT="${1:-${BASE_MAIN_OUT:-reports/main_results_ragas_fixed120_login_20260416_115722}}"
RESUME_COMPLETED_OUT="${2:-${RESUME_COMPLETED_OUT:-reports/main_results_ragas_fixed120_resume_completed_20260418}}"
SUPPLEMENT_MAIN_OUT="${3:-${SUPPLEMENT_MAIN_OUT:-reports/main_results_ragas_fixed120_missing2_20260418}}"
MERGED_MAIN_OUT="${4:-${MERGED_MAIN_OUT:-reports/main_results_ragas_fixed120_controlled8_merged_20260418}}"
OUT_DIR="${5:-${OUT_DIR:-reports/category_summary_fixed120_real_20260418}}"

if [ ! -d "$BASE_MAIN_OUT" ]; then
  echo "Base main dir not found: $BASE_MAIN_OUT" >&2
  exit 2
fi

if [ ! -d "$RESUME_COMPLETED_OUT" ]; then
  echo "Resume completed dir not found: $RESUME_COMPLETED_OUT" >&2
  exit 2
fi

if [ ! -d "$SUPPLEMENT_MAIN_OUT" ]; then
  echo "Supplement main dir not found: $SUPPLEMENT_MAIN_OUT" >&2
  exit 2
fi

<local_path_removed>
  --input-dir "$BASE_MAIN_OUT" \
  --input-dir "$RESUME_COMPLETED_OUT" \
  --input-dir "$SUPPLEMENT_MAIN_OUT" \
  --output-dir "$MERGED_MAIN_OUT"

mkdir -p "$OUT_DIR"

<local_path_removed>
  --main-dir "$MERGED_MAIN_OUT" \
  --out-dir "$OUT_DIR"
