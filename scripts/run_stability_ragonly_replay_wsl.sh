#!/bin/bash
set -euo pipefail

cd <local_path_removed>

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export BASE_URL="${BASE_URL:-https://api.vectorengine.cn/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$BASE_URL}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$BASE_URL}"

export PGHOST="${PGHOST:-127.0.0.1}"
export PGPORT="${PGPORT:-55440}"
export PGDATABASE="${PGDATABASE:-ragkb}"
export PGUSER="${PGUSER:-ragkb}"
export PGPASSWORD="${PGPASSWORD:-ragkb}"
export PGTABLE="${PGTABLE:-rag_documents_bge_m3}"

export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export ANSWER_MODE="${ANSWER_MODE:-openai}"
export EVIDENCE_MODE="${EVIDENCE_MODE:-none}"
export MIN_SCORE="${MIN_SCORE:-0.15}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM='false'

export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-bge_m3}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-<local_path_removed>"
export EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
export BGE_M3_USE_FP16="${BGE_M3_USE_FP16:-auto}"
export BGE_M3_BATCH_SIZE="${BGE_M3_BATCH_SIZE:-8}"
export BGE_M3_MAX_LENGTH="${BGE_M3_MAX_LENGTH:-8192}"
export RERANKER_MODEL="${RERANKER_MODEL:-<local_path_removed>"

MAIN_DIR="${1:-${MAIN_DIR:-}}"
SOURCE_SAMPLING_ROOT="${2:-${SOURCE_SAMPLING_ROOT:-}}"
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-reports/stability_ragonly_replay_$(date +%Y%m%d_%H%M%S)}}"

if [ -z "$MAIN_DIR" ] || [ -z "$SOURCE_SAMPLING_ROOT" ]; then
  echo "Usage: $0 <main_dir> <source_sampling_root> [output_root]" >&2
  exit 2
fi

<local_path_removed>
  --main-dir "$MAIN_DIR" \
  --source-sampling-root "$SOURCE_SAMPLING_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --eval-mode ragas

