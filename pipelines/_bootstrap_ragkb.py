from __future__ import annotations

import sys
from pathlib import Path


def ensure_ragkb_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ragkb_src = repo_root.parent / "llm-rag-knowledge-base" / "src"
    if ragkb_src.exists() and str(ragkb_src) not in sys.path:
        sys.path.insert(0, str(ragkb_src))
