#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
RAGKB_ROOT="$WORKSPACE_ROOT/llm-rag-knowledge-base"
BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_wsl_env.sh"
VENV_DIR="$REPO_ROOT/.venv_wsl"
VENV_PY="$VENV_DIR/bin/python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is required inside WSL."
  exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
  bash "$BOOTSTRAP_SCRIPT"
fi

load_env_keys() {
  local file="$1"
  shift
  if [[ ! -f "$file" ]]; then
    return
  fi
  while IFS='=' read -r raw_key raw_value; do
    raw_key="${raw_key%%$'\r'}"
    raw_value="${raw_value%%$'\r'}"
    [[ -z "$raw_key" ]] && continue
    [[ "$raw_key" =~ ^[[:space:]]*# ]] && continue
    for wanted in "$@"; do
      if [[ "$raw_key" == "$wanted" ]]; then
        export "$raw_key=$raw_value"
        break
      fi
    done
  done <"$file"
}

load_env_keys "$REPO_ROOT/.env" OPENAI_API_KEY BASE_URL OPENAI_BASE_URL OPENAI_API_BASE LLM_MODEL
load_env_keys "$RAGKB_ROOT/.env" OPENAI_API_KEY BASE_URL OPENAI_BASE_URL OPENAI_API_BASE LLM_MODEL

export PGHOST=127.0.0.1
export PGPORT="${PGPORT:-5432}"
export PGDATABASE="${PGDATABASE:-ragkb}"
export PGUSER="${PGUSER:-ragkb}"
export PGPASSWORD="${PGPASSWORD:-ragkb}"
export PGTABLE="${PGTABLE:-rag_documents_bge_m3}"
export PGCONNECT_TIMEOUT="${PGCONNECT_TIMEOUT:-5}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-bge_m3}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-$WORKSPACE_ROOT/models/bge-m3-local}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
export ANSWER_MODE="${ANSWER_MODE:-openai}"
export EVIDENCE_MODE="${EVIDENCE_MODE:-none}"
export RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid}"
export RERANKER_ENABLED="${RERANKER_ENABLED:-1}"
export BGE_M3_USE_FP16="${BGE_M3_USE_FP16:-auto}"
export BGE_M3_BATCH_SIZE="${BGE_M3_BATCH_SIZE:-8}"
export BGE_M3_MAX_LENGTH="${BGE_M3_MAX_LENGTH:-8192}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$REPO_ROOT"

if [[ "${RUN_PROMPTFOO_IN_WSL:-0}" == "1" ]] && command -v npx >/dev/null 2>&1; then
  npx promptfoo eval -c configs/promptfoo.yaml || true
fi

"$VENV_PY" -m pipelines.compare --data-path data/fbtp_eval.jsonl --output-dir reports/latest --eval-mode ragas "$@"
