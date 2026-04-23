from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv

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


def _get_bootstrap_module():
    from db_rag import bootstrap

    return bootstrap


def _runtime_assets_ready() -> bool:
    DbRagService = _get_db_rag_service_class()
    readiness = DbRagService(llm=object()).readiness()
    return bool(readiness.get("ready"))


def ensure_assets_ready(*, rebuild_if_missing: bool = True) -> dict[str, Any]:
    DbRagService = _get_db_rag_service_class()
    bootstrap = _get_bootstrap_module()

    readiness = DbRagService(llm=object()).readiness()
    if readiness.get("ready"):
        return readiness

    if rebuild_if_missing:
        bootstrap.rebuild()
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


def run_query(
    question: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model_name: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    rebuild_if_missing: bool = True,
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
    return service.execute_sql_flow(question)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    question = _resolve_query(args)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")

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
        rebuild_if_missing=not args.no_rebuild,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
