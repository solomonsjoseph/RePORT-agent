from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv

from db_rag.config import (
    SUPPORTED_DB_RAG_RERANKER_MODELS,
    resolve_db_rag_embedding_model,
)
from llm_vllm import build_llm, detect_vllm_model
from utils.streamlit_config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PROVIDER_OPTIONS,
)


def _get_db_rag_service_class():
    from db_rag.service import DbRagService

    return DbRagService


def _get_build_index_module():
    from db_rag import build_index

    return build_index


def _runtime_assets_ready() -> bool:
    DbRagService = _get_db_rag_service_class()
    readiness = DbRagService(llm=object()).readiness()
    return bool(readiness.get("ready"))


def ensure_assets_ready(*, rebuild_if_missing: bool = True) -> dict[str, Any]:
    DbRagService = _get_db_rag_service_class()
    build_index = _get_build_index_module()

    readiness = DbRagService(llm=object()).readiness()
    if readiness.get("ready"):
        return readiness

    if rebuild_if_missing:
        build_index.rebuild()
        readiness = DbRagService(llm=object()).readiness()

    if not readiness.get("ready"):
        raise RuntimeError(str(readiness.get("message") or "DB-RAG runtime assets are not ready."))
    return readiness


def _default_model_name(provider: str, base_url: str) -> str:
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    if provider == "vllm":
        return detect_vllm_model(base_url)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def _resolve_api_key(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "")
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY", "")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return ""


def build_runtime_llm(
    *,
    provider: str,
    model_name: str | None,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
):
    resolved_model = model_name or _default_model_name(provider, base_url)
    return build_llm(
        model_name=resolved_model,
        temperature=temperature,
        top_p=top_p,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
    )


def resolve_query_llm_label(*, provider: str, model_name: str | None, base_url: str) -> str:
    provider_labels = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "vllm": "vLLM",
    }
    provider_label = provider_labels.get(provider, str(provider))
    resolved_model = model_name or _default_model_name(provider, base_url)
    return f"{provider_label} / {resolved_model}"


def run_query(
    question: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model_name: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    reranker_model: str | None = None,
    rebuild_if_missing: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    DbRagService = _get_db_rag_service_class()
    ensure_assets_ready(rebuild_if_missing=rebuild_if_missing)
    llm = build_runtime_llm(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
    )
    service = DbRagService(llm=llm)
    kwargs: dict[str, Any] = {"debug": debug}
    if reranker_model is not None:
        kwargs["reranker_model"] = reranker_model
    return service.execute_sql_flow(question, **kwargs)


def _print_debug(result: dict[str, Any]) -> None:
    debug = result.get("debug") or {}
    if not debug:
        return
    print("\nRetrieval summary:")
    print(f"Retrieved tables: {debug.get('retrieved_tables', [])}")
    print(f"Retrieved columns: {debug.get('retrieved_columns', [])}")
    print("\nSQL preparation:")
    print(f"Question: {debug.get('question', '')}")
    print(f"Tables: {debug.get('sql_tables', [])}")
    print(f"Columns: {debug.get('sql_columns', [])}")


def _print_result(result: dict[str, Any]) -> None:
    print(f"Answer: {result.get('answer', '')}")
    sql = str(result.get("sql", "") or "").strip()
    if sql:
        print("\nSQL:")
        print(sql)

    dataframe = result.get("dataframe")
    if dataframe is not None:
        print("\nResult:")
        print(dataframe.to_string(index=False))

    source_tables = result.get("source_tables")
    if source_tables:
        print("\nSource tables:")
        print(", ".join(source_tables))


def _print_runtime_banner(
    *,
    question: str,
    indexing_model: str,
    reranker_model: str | None,
    provider: str,
    model_name: str | None,
    base_url: str,
) -> None:
    print("\nRunning quick test:")
    print(f"Question: {question}")
    print(f"Indexing model: {indexing_model}")
    if reranker_model:
        print(f"Reranker: {reranker_model}")
    else:
        print("Reranker: none (ChromaDB ordering)")
        available = ", ".join(SUPPORTED_DB_RAG_RERANKER_MODELS)
        print("To enable reranking, pass --reranker <model>.")
        print(f"Available reranker models: {available}")
    print(f"Query LLM: {resolve_query_llm_label(provider=provider, model_name=model_name, base_url=base_url)}")
    print()


def _resolve_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query.strip()
    while True:
        query = input("> ").strip()
        if query:
            return query


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m db_rag.quick_test",
        description="Quick DB-RAG demo for the current repo.",
    )
    parser.add_argument("query", nargs="?", help="Question to ask the DB-RAG demo.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_OPTIONS,
        default=DEFAULT_PROVIDER,
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name. If omitted, the script uses the provider default.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for vLLM providers.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Falls back to environment variables when omitted.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="LLM top-p.",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Fail instead of rebuilding when DB-RAG assets are missing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieval and SQL-generation details for the DB-RAG demo flow.",
    )
    parser.add_argument(
        "--reranker",
        choices=SUPPORTED_DB_RAG_RERANKER_MODELS,
        default=None,
        help="Optional column reranker. Omit to preserve ChromaDB ordering.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    question = _resolve_query(args)
    api_key = _resolve_api_key(args.provider, args.api_key)
    indexing_model = resolve_db_rag_embedding_model()

    _print_runtime_banner(
        question=question,
        indexing_model=indexing_model,
        reranker_model=args.reranker,
        provider=args.provider,
        model_name=args.model,
        base_url=args.base_url,
    )

    if not _runtime_assets_ready():
        print("DB-RAG assets are missing. Rebuilding runtime assets...")

    result = run_query(
        question,
        provider=args.provider,
        model_name=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        reranker_model=args.reranker,
        rebuild_if_missing=not args.no_rebuild,
        debug=args.debug,
    )
    if args.debug:
        _print_debug(result)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
