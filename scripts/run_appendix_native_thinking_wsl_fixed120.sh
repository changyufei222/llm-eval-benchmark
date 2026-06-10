#!/bin/bash
set -euo pipefail

cd <local_path_removed>

set -a
[ -f ./.env ] && source ./.env
set +a

export BASE_URL="${BASE_URL:-https://api.vectorengine.cn/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$BASE_URL}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$BASE_URL}"

export PGHOST='localhost'
export PGPORT='5432'
export PGDATABASE='ragkb'
export PGUSER='ragkb'
export PGPASSWORD='ragkb'
export PGTABLE='rag_documents_bge_m3'

export LLM_PROVIDER='openai'
export LLM_MODEL='DeepSeek-V3.2'
export ANSWER_MODE='openai'
