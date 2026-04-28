from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import load_dotenv

from .config import DEFAULT_OPENROUTER_BASE_URL, SUPPORTED_DB_RAG_RERANKER_MODELS


class OpenAIEmbeddingFunction:
    _MAX_EMPTY_DATA_RETRIES = 3

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

    def _create_embedding_batch(self, batch: list[str]) -> list[list[float]]:
        for _attempt in range(self._MAX_EMPTY_DATA_RETRIES):
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                **self._embedding_create_kwargs,
            )
            data = getattr(response, "data", None)
            if data:
                return [item.embedding for item in data]
        preview = batch[0][:120] if batch else ""
        raise ValueError(
            f"No embedding data received for model {self.config_model} "
            f"after {self._MAX_EMPTY_DATA_RETRIES} attempts. Query preview: {preview!r}"
        )

    def __call__(self, input: str | list[str]) -> list[list[float]]:
        normalized_input = self._normalize_input(input)
        embeddings: list[list[float]] = []
        for start in range(0, len(normalized_input), 100):
            batch = normalized_input[start : start + 100]
            embeddings.extend(self._create_embedding_batch(batch))
        return embeddings


class OpenAIReranker:
    def __init__(self, model: str | None):
        load_dotenv()

        resolved_model = str(model or "").strip()
        if not resolved_model:
            self.model = ""
            self.config_model = None
            self.api_key = ""
            self.base_url = ""
            return
        if resolved_model not in SUPPORTED_DB_RAG_RERANKER_MODELS:
            supported = ", ".join(SUPPORTED_DB_RAG_RERANKER_MODELS)
            raise ValueError(f"Unsupported DB-RAG reranker model '{resolved_model}'. Supported values: {supported}.")

        api_key = str(os.getenv("DB_RAG_OPENROUTER_API_KEY", "") or "").strip()
        if not api_key:
            raise ValueError("DB_RAG_OPENROUTER_API_KEY is required for OpenRouter reranking.")

        self.model = resolved_model
        self.config_model = resolved_model
        self.api_key = api_key
        self.base_url = str(
            os.getenv("DB_RAG_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL) or DEFAULT_OPENROUTER_BASE_URL
        ).strip().rstrip("/")

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self.model or not documents:
            return [0.0] * len(documents)

        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        request = urllib_request.Request(
            url=f"{self.base_url}/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter rerank request failed for model {self.config_model}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"OpenRouter rerank request failed for model {self.config_model}: {exc}") from exc

        results = body.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Unexpected rerank response for model {self.config_model}: missing results.")

        scores = [0.0] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                continue
            scores[index] = float(item.get("relevance_score", 0.0) or 0.0)
        return scores


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
