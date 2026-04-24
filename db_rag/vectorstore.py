from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import DEFAULT_OPENROUTER_BASE_URL


class OpenAIEmbeddingFunction:
    def __init__(self, model: str):
        load_dotenv()
        from openai import OpenAI

        resolved_model = str(model or "").strip()
        api_model = resolved_model.split("/", 1)[1] if resolved_model.startswith("OpenAI/") else resolved_model
        client_kwargs: dict[str, str] = {}
        self._embedding_create_kwargs: dict[str, str] = {}
        if resolved_model.startswith("Qwen/"):
            api_key = str(os.getenv("DB_RAG_OPENROUTER_API_KEY", "") or "").strip()
            if not api_key:
                raise ValueError("DB_RAG_OPENROUTER_API_KEY is required for Qwen embeddings.")
            client_kwargs["api_key"] = api_key
            client_kwargs["base_url"] = str(
                os.getenv("DB_RAG_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL) or DEFAULT_OPENROUTER_BASE_URL
            ).strip()
            api_model = resolved_model.lower()
            self._embedding_create_kwargs["encoding_format"] = "float"
        self.client = OpenAI(**client_kwargs)
        self.model = api_model
        self.config_model = resolved_model

    @staticmethod
    def name() -> str:
        return "openai"

    @staticmethod
    def _normalize_input(input: str | list[str]) -> list[str]:
        if isinstance(input, str):
            return [input]
        return list(input)

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        return self.__call__(input)

    def __call__(self, input: str | list[str]) -> list[list[float]]:
        normalized_input = self._normalize_input(input)
        embeddings: list[list[float]] = []
        for start in range(0, len(normalized_input), 100):
            batch = normalized_input[start : start + 100]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                **self._embedding_create_kwargs,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


def build_chroma(
    table_chunks: list[dict[str, object]],
    column_chunks: list[dict[str, object]],
    *,
    model: str,
    chroma_dir: Path,
) -> None:
    import chromadb

    if chroma_dir.exists():
        for child in chroma_dir.iterdir():
            if child.is_file():
                child.unlink()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    for collection_name in ("table_summaries", "column_chunks"):
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    ef = OpenAIEmbeddingFunction(model=model)
    table_collection = client.create_collection("table_summaries", embedding_function=ef)
    column_collection = client.create_collection("column_chunks", embedding_function=ef)
    table_collection.add(
        ids=[chunk["id"] for chunk in table_chunks],
        documents=[chunk["text"] for chunk in table_chunks],
        metadatas=[chunk["metadata"] for chunk in table_chunks],
    )
    column_collection.add(
        ids=[chunk["id"] for chunk in column_chunks],
        documents=[chunk["text"] for chunk in column_chunks],
        metadatas=[chunk["metadata"] for chunk in column_chunks],
    )


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
