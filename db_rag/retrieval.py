from __future__ import annotations

from typing import Any

from utils.llm_response import coerce_text_content


PAIRED_FORMS = {
    "Off Study Form for Cohort A (Form F99A)": "Form 1A - Index Case Screening",
    "Off Study Form for Cohort B (Form 99B)": "Form 1B - Household Contact Screening Form",
    "Final Outcome Determination Form - Cohort A (Active Pulmonary TB)": "Form 1A - Index Case Screening",
    "Final Outcome Determination Form - Cohort B (Household Contacts)": "Form 1B - Household Contact Screening Form",
}


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


def retrieve_context_records(
    llm: Any,
    table_collection: Any,
    column_collection: Any,
    question: str,
    *,
    table_k: int = 4,
    column_k: int = 12,
    debug: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sub_queries = decompose_query(llm, question)
    if debug and len(sub_queries) > 1:
        print("\nQuery decomposition:")
        for sub_query in sub_queries:
            print(f"  -> {sub_query}")
    if len(sub_queries) <= 1:
        if debug:
            print("\nSingle-concept query, skipping decomposition")
        return retrieve_single_query(
            table_collection,
            column_collection,
            question,
            table_k=table_k,
            column_k=column_k,
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
    selected_tables = set(merged_tables)
    if debug:
        print(f"\nMerged tables: {len(merged_tables)}")
        for entry in merged_tables.values():
            print(f"  {entry['table']}")

    merged_columns: dict[tuple[str, str], dict[str, str]] = {}
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
    if debug:
        print(f"Merged column candidates: {len(merged_columns)}")
        for entry in merged_columns.values():
            print(f"  {entry['table']}.{entry['column']}")

    return list(merged_tables.values()), list(merged_columns.values())
