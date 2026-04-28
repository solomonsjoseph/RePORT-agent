from __future__ import annotations

from typing import Any

from utils.llm_response import coerce_text_content

from .vectorstore import OpenAIReranker


PAIRED_FORMS = {
    "Off Study Form for Cohort A (Form F99A)": "Form 1A - Index Case Screening",
    "Off Study Form for Cohort B (Form 99B)": "Form 1B - Household Contact Screening Form",
    "Final Outcome Determination Form - Cohort A (Active Pulmonary TB)": "Form 1A - Index Case Screening",
    "Final Outcome Determination Form - Cohort B (Household Contacts)": "Form 1B - Household Contact Screening Form",
}
RESERVED_PER_CONCEPT = 2


def decompose_query(llm: Any, question: str) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a query decomposer for a clinical database. Extract the distinct concepts "
                        "from the question and return each as a short search phrase (2-5 words). "
                        "One per line. No numbering, no explanation, no commentary."
                    )
                ),
                HumanMessage(content=question),
            ]
        )
    except Exception:
        return []

    phrases: list[str] = []
    for raw_line in coerce_text_content(getattr(response, "content", "")).splitlines():
        line = raw_line.strip().lstrip("- ").lstrip("0123456789.)").strip()
        if line and len(line) < 50 and not line.lower().startswith("question"):
            phrases.append(line)
    return phrases


def retrieve_single_query(
    table_collection: Any,
    column_collection: Any,
    query: str,
    *,
    table_k: int = 4,
    column_k: int = 12,
    debug: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    table_result = table_collection.query(
        query_texts=[query],
        n_results=table_k,
        include=["documents", "metadatas"],
    )
    tables: list[dict[str, str]] = []
    for document, metadata in zip(table_result["documents"][0], table_result["metadatas"][0]):
        tables.append({"table": metadata["table"], "text": document})
    if debug:
        print("\nTable retrieval:")
        for entry in tables:
            print(f"  {entry['table']}")

    selected_tables = {entry["table"] for entry in tables}
    column_result = column_collection.query(
        query_texts=[query],
        n_results=column_k,
        include=["documents", "metadatas"],
    )
    columns: list[dict[str, str]] = []
    for document, metadata in zip(column_result["documents"][0], column_result["metadatas"][0]):
        if metadata["table"] not in selected_tables:
            continue
        columns.append({"table": metadata["table"], "column": metadata["column"], "text": document})
    if debug:
        print("\nColumn retrieval:")
        for entry in columns:
            print(f"  {entry['table']}.{entry['column']}")

    return tables, columns


def rerank_columns(
    query: str,
    column_hits: list[dict[str, str]],
    *,
    reranker_model: str | None,
    top_k: int,
    debug: bool = False,
) -> list[dict[str, str]]:
    if not column_hits:
        return []

    if not reranker_model:
        if debug:
            print("\nReranking disabled, using ChromaDB ordering")
        return column_hits[:top_k]

    reranker = OpenAIReranker(model=reranker_model)
    scores = reranker.rerank(query, [hit["text"] for hit in column_hits])
    scored_hits = sorted(zip(scores, column_hits), key=lambda item: item[0], reverse=True)

    if debug:
        print("\nReranker scores (after reranking):")
        for score, hit in scored_hits[:top_k]:
            print(f"  score={score:.4f}  {hit['table']}.{hit['column']}")

    return [hit for _, hit in scored_hits[:top_k]]


def inject_paired_forms(merged_tables: dict[str, dict[str, str]], table_collection: Any, *, debug: bool = False) -> None:
    for table_name in list(merged_tables):
        paired_form = PAIRED_FORMS.get(table_name)
        if not paired_form or paired_form in merged_tables:
            continue
        result = table_collection.query(
            query_texts=[paired_form],
            n_results=1,
            include=["documents", "metadatas"],
        )
        for document, metadata in zip(result["documents"][0], result["metadatas"][0]):
            if metadata["table"] == paired_form:
                merged_tables[paired_form] = {"table": metadata["table"], "text": document}
                if debug:
                    print(f"Injected paired form: {paired_form}")
                break


def _inject_required_tables(
    merged_tables: dict[str, dict[str, str]],
    table_collection: Any,
    required_tables: list[str],
) -> None:
    for table_name in required_tables:
        if table_name in merged_tables:
            continue
        result = table_collection.query(
            query_texts=[table_name],
            n_results=1,
            include=["documents", "metadatas"],
        )
        for document, metadata in zip(result["documents"][0], result["metadatas"][0]):
            if metadata["table"] == table_name:
                merged_tables[table_name] = {"table": metadata["table"], "text": document}
                break


def _drop_excluded_tables(
    merged_tables: dict[str, dict[str, str]],
    excluded_tables: list[str],
) -> None:
    for table_name in excluded_tables:
        merged_tables.pop(table_name, None)


def retrieve_context_records(
    llm: Any,
    table_collection: Any,
    column_collection: Any,
    question: str,
    *,
    table_k: int = 4,
    column_k: int = 12,
    reranker_model: str | None = None,
    debug: bool = False,
    required_tables: list[str] | None = None,
    excluded_tables: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sub_queries = decompose_query(llm, question)
    if debug and len(sub_queries) > 1:
        print("\nQuery decomposition:")
        for sub_query in sub_queries:
            print(f"  -> {sub_query}")
    if len(sub_queries) <= 1:
        if debug:
            print("\nSingle-concept query, skipping decomposition")
        tables, columns = retrieve_single_query(
            table_collection,
            column_collection,
            question,
            table_k=table_k,
            column_k=column_k,
            debug=debug,
        )
        merged_tables = {entry["table"]: entry for entry in tables}
        _inject_required_tables(merged_tables, table_collection, list(required_tables or []))
        _drop_excluded_tables(merged_tables, list(excluded_tables or []))
        selected_tables = set(merged_tables)
        columns = [entry for entry in columns if entry["table"] in selected_tables]
        return list(merged_tables.values()), rerank_columns(
            question,
            columns,
            reranker_model=reranker_model,
            top_k=column_k,
            debug=debug,
        )

    merged_tables: dict[str, dict[str, str]] = {}
    for sub_query in sub_queries:
        table_result = table_collection.query(
            query_texts=[sub_query],
            n_results=table_k,
            include=["documents", "metadatas"],
        )
        for document, metadata in zip(table_result["documents"][0], table_result["metadatas"][0]):
            merged_tables.setdefault(metadata["table"], {"table": metadata["table"], "text": document})

    inject_paired_forms(merged_tables, table_collection, debug=debug)
    _inject_required_tables(merged_tables, table_collection, list(required_tables or []))
    _drop_excluded_tables(merged_tables, list(excluded_tables or []))
    selected_tables = set(merged_tables)
    if debug:
        print(f"\nMerged tables: {len(merged_tables)}")
        for entry in merged_tables.values():
            print(f"  {entry['table']}")

    merged_columns: dict[tuple[str, str], dict[str, str]] = {}
    column_to_subquery: dict[tuple[str, str], str] = {}
    for sub_query in sub_queries:
        column_result = column_collection.query(
            query_texts=[sub_query],
            n_results=column_k,
            include=["documents", "metadatas"],
        )
        for document, metadata in zip(column_result["documents"][0], column_result["metadatas"][0]):
            if metadata["table"] not in selected_tables:
                continue
            key = (metadata["table"], metadata["column"])
            merged_columns.setdefault(
                key,
                {"table": metadata["table"], "column": metadata["column"], "text": document},
            )
            column_to_subquery.setdefault(key, sub_query)
    if debug:
        print(f"Merged column candidates: {len(merged_columns)}")
        for entry in merged_columns.values():
            print(f"  {entry['table']}.{entry['column']}")

    all_reranked = rerank_columns(
        question,
        list(merged_columns.values()),
        reranker_model=reranker_model,
        top_k=len(merged_columns),
        debug=debug,
    )

    reserved: dict[tuple[str, str], dict[str, str]] = {}
    for sub_query in sub_queries:
        sub_query_hits = [
            hit
            for hit in all_reranked
            if column_to_subquery.get((hit["table"], hit["column"])) == sub_query
        ]
        for hit in sub_query_hits[:RESERVED_PER_CONCEPT]:
            reserved[(hit["table"], hit["column"])] = hit

    final_columns = list(reserved.values())
    seen = set(reserved)
    for hit in all_reranked:
        if len(final_columns) >= column_k:
            break
        key = (hit["table"], hit["column"])
        if key in seen:
            continue
        final_columns.append(hit)
        seen.add(key)

    if debug:
        print(f"\nFinal columns ({len(final_columns)}) with guaranteed concept coverage:")
        for hit in final_columns:
            key = (hit["table"], hit["column"])
            source = column_to_subquery.get(key, "?")
            marker = " [reserved]" if key in reserved else ""
            print(f"  {hit['table']}.{hit['column']} (from: {source}){marker}")

    return list(merged_tables.values()), final_columns
