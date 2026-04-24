from __future__ import annotations

import hashlib
import json
import re
from typing import Any


DB_RAG_CONTEXT_FALLBACK_ANSWER = "I could not answer from the retrieved DB-RAG context."
DB_RAG_CONTEXT_FALLBACK_RATIONALE = "Invalid structured response from the model."
_DANGEROUS_SQL = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "CREATE", "REPLACE")


def extract_sql(text: str) -> str:
    text = str(text or "").strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for prefix in ("SELECT", "WITH"):
        idx = text.upper().find(prefix)
        if idx >= 0:
            return text[idx:].strip()
    return text


def parse_json_object(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def default_selection_id(question: str, table_names: list[str], columns: list[dict[str, str]], feedback_history: list[dict[str, Any]]) -> str:
    payload = {
        "question": question,
        "tables": table_names,
        "columns": columns,
        "feedback_history": feedback_history,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sel-{digest[:12]}"


def validate_sql(sql: str) -> tuple[bool, str | None]:
    sql_upper = str(sql or "").strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH")):
        return False, "Only read-only SELECT/WITH SQL is allowed."
    if any(keyword in sql_upper for keyword in _DANGEROUS_SQL):
        return False, "SQL contains disallowed mutating or DDL keywords."

    try:
        import sqlglot

        sqlglot.parse_one(sql, dialect="duckdb")
    except ModuleNotFoundError:
        return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)
    return True, None
