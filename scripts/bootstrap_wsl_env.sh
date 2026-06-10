#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
RAGKB_ROOT="$WORKSPACE_ROOT/llm-rag-knowledge-base"
VENV_DIR="$REPO_ROOT/.venv_wsl"
VENV_PY="$VENV_DIR/bin/python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is required inside WSL."
  exit 1
fi

RECREATE="${RECREATE:-0}"
if [[ "$RECREATE" == "1" || ! -x "$VENV_PY" ]]; then
  rm -rf "$VENV_DIR"
  mkdir -p "$VENV_DIR/lib64"
  "$PYTHON_BIN" -m venv --copies "$VENV_DIR"
fi

"$VENV_PY" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
"$VENV_PY" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu"
"$VENV_PY" -m pip install --disable-pip-version-check \
  "ragas==0.4.3" \
  "datasets==4.8.4" \
  "langchain-core==1.2.22" \
  "langchain-openai==1.1.12" \
  "openai==2.29.0" \
  "pandas==3.0.1" \
  "numpy==2.4.3" \
  "matplotlib==3.10.8" \
  "tabulate==0.10.0" \
  "python-dotenv==1.2.2" \
  "FlagEmbedding==1.3.5" \
  "transformers==4.57.6" \
  "huggingface_hub==0.36.2" \
  "accelerate==1.13.0" \
  "sentence-transformers==5.3.0" \
  "sentencepiece==0.2.1" \
  "regex==2026.2.28" \
  "psycopg[binary]==3.3.3" \
  "pgvector==0.3.6" \
  "python-docx==1.2.0" \
  "pypdf==6.9.2" \
  "lxml==6.0.2" \
  "instructor==1.14.5" \
  "scikit-network==0.33.5" \
  "appdirs==1.4.4" \
  "diskcache==5.6.3" \
  "nest-asyncio==1.6.0"

"$VENV_PY" -m pip install --disable-pip-version-check -e "$RAGKB_ROOT" -e "$REPO_ROOT"

echo "Created WSL benchmark environment:"
echo "  $VENV_DIR"
echo "Interpreter:"
echo "  $VENV_PY"
