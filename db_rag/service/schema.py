from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any

from db_rag.config import SCHEMA_DIR


@lru_cache(maxsize=1)
def _schema_column_catalog() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    by_column: dict[str, list[dict[str, str]]] = {}
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table = str(schema.get("form_name", "") or "").strip()
        variables = dict(schema.get("variables", {}) or {})
        if not table:
            continue
        for column, info in variables.items():
            column_name = str(column or "").strip()
            if not column_name:
                continue
            entry = {
                "table": table,
                "column": column_name,
                "description": str((info or {}).get("description", "") or "").strip(),
            }
            by_pair[(table, column_name)] = entry
            by_column.setdefault(column_name, []).append(entry)
    return by_pair, by_column


@lru_cache(maxsize=1)
def _schema_variable_catalog() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    by_column: dict[str, list[dict[str, Any]]] = {}
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table = str(schema.get("form_name", "") or "").strip()
        variables = dict(schema.get("variables", {}) or {})
        if not table:
            continue
        for column, info in variables.items():
            column_name = str(column or "").strip()
            if not column_name:
                continue
            variable_info = dict(info or {})
            entry = {
                "table": table,
                "column": column_name,
                "description": str(variable_info.get("description", "") or "").strip(),
                "values": variable_info.get("values"),
                "depends_on": variable_info.get("depends_on"),
                "condition": variable_info.get("condition"),
                "section_context": variable_info.get("section_context"),
            }
            by_pair[(table, column_name)] = entry
            by_column.setdefault(column_name, []).append(entry)
    return by_pair, by_column


def _normalize_table_label(table: str) -> str:
    text = str(table or "").strip().lower()
    if not text:
        return ""
    text = text.split(":", 1)[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _lookup_schema_column(table: str, column: str) -> dict[str, str] | None:
    by_pair, _by_column = _schema_column_catalog()
    return by_pair.get((str(table or "").strip(), str(column or "").strip()))


def _lookup_schema_variable_metadata(table: str, column: str) -> dict[str, Any] | None:
    by_pair, by_column = _schema_variable_catalog()
    raw_table = str(table or "").strip()
    raw_column = str(column or "").strip()
    if not raw_column:
        return None

    exact = by_pair.get((raw_table, raw_column))
    if exact is not None:
        return dict(exact)

    normalized_table = _normalize_table_label(raw_table)
    if normalized_table:
        matches = [
            entry
            for (entry_table, entry_column), entry in by_pair.items()
            if entry_column == raw_column
            and (
                _normalize_table_label(entry_table).startswith(normalized_table)
                or normalized_table.startswith(_normalize_table_label(entry_table))
            )
        ]
        if len(matches) == 1:
            return dict(matches[0])

    column_matches = list(by_column.get(raw_column) or [])
    if len(column_matches) == 1:
        return dict(column_matches[0])
    return None


def _schema_table_names() -> set[str]:
    by_pair, _ = _schema_column_catalog()
    return {table for table, _column in by_pair}


def _resolve_explicit_schema_mentions(
    question: str,
    feedback_history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    by_pair, by_column = _schema_column_catalog()
    texts = [str(question or "")]
    texts.extend(str(item.get("feedback") or "") for item in feedback_history if isinstance(item, dict))
    haystack = "\n".join(texts)
    haystack_lower = haystack.lower()

    matched: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for (table, column), entry in by_pair.items():
        token = f"{table}.{column}".lower()
        if token in haystack_lower and (table, column) not in seen_pairs:
            matched.append(dict(entry))
            seen_pairs.add((table, column))

    for column_name, entries in by_column.items():
        if len(entries) != 1:
            continue
        if re.search(rf"\b{re.escape(column_name)}\b", haystack, re.IGNORECASE):
            entry = entries[0]
            pair = (entry["table"], entry["column"])
            if pair not in seen_pairs:
                matched.append(dict(entry))
                seen_pairs.add(pair)

    return matched
