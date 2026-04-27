from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "local_data" / "db_rag_source"
SCHEMA_DIR = SOURCE_ROOT / "reviewed_annotated_json_files"
EXCEL_DIR = SOURCE_ROOT / "filtered_excel_files"
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "db_rag"
INDEX_ROOT = RUNTIME_ROOT / "indexes"
MANIFEST_ROOT = RUNTIME_ROOT / "manifests"
CHROMA_DIR = RUNTIME_ROOT / "chroma_db"
DUCKDB_PATH = RUNTIME_ROOT / "report.duckdb"
MANIFEST_PATH = RUNTIME_ROOT / "manifest.json"
EMBEDDING_MODEL = "OpenAI/text-embedding-3-small"
SUPPORTED_DB_RAG_EMBEDDING_MODELS = (
    "OpenAI/text-embedding-3-small",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
)
SUPPORTED_DB_RAG_RERANKER_MODELS = (
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def embedding_model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def chroma_dir_for_model(model: str) -> Path:
    return INDEX_ROOT / embedding_model_slug(model)


def manifest_path_for_model(model: str) -> Path:
    return MANIFEST_ROOT / f"{embedding_model_slug(model)}.json"


def load_manifest_for_model(model: str) -> dict[str, Any]:
    path = manifest_path_for_model(model)
    return json.loads(path.read_text(encoding="utf-8"))


def env_path_for_project(project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / ".env"


def resolve_db_rag_embedding_model() -> str:
    load_dotenv()
    model = str(os.getenv("DB_RAG_EMBEDDING_MODEL", "") or "").strip()
    if not model:
        raise ValueError("DB_RAG_EMBEDDING_MODEL is not set. Set it in .env or export it in your shell.")
    if model not in SUPPORTED_DB_RAG_EMBEDDING_MODELS:
        supported = ", ".join(SUPPORTED_DB_RAG_EMBEDDING_MODELS)
        raise ValueError(f"Unsupported DB_RAG_EMBEDDING_MODEL '{model}'. Supported values: {supported}.")
    return model
