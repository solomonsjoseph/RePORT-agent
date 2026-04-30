from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .config import PRIMARY_JOIN_KEY_ALIAS, SECONDARY_JOIN_KEY_ALIAS

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


def is_unanswerable_response(text: str) -> bool:
    return str(text or "").strip().upper().startswith("UNANSWERABLE")


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


def build_sql_policy_text() -> str:
    return (
        "JOIN HINTS:\n"
        "Use join keys from the table context.\n"
        f"Treat canonical join-key labels like {PRIMARY_JOIN_KEY_ALIAS} and {SECONDARY_JOIN_KEY_ALIAS} "
        "as semantic join categories, while using exact source column names in SQL.\n"
        "If one table suffices, do NOT invent joins.\n\n"
        "RULES:\n"
        "- Use DuckDB SQL.\n"
        "- Use ILIKE for case-insensitive string comparisons.\n"
        "- Prefer the stored/sample-value representation over coded allowed values when they differ.\n"
        "- Values of 99, 999, or 9999 are often missing data codes and should be excluded unless the question is explicitly about missingness.\n"
        "- If a column profile shows a high NULL rate, add IS NOT NULL where appropriate."
    )


def _mask_sql_literals_identifiers_and_comments(sql: str) -> str:
    masked: list[str] = []
    quote_char: str | None = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote_char:
            masked.append(" ")
            if char == quote_char:
                if i + 1 < len(sql) and sql[i + 1] == quote_char:
                    masked.append(" ")
                    i += 2
                    continue
                quote_char = None
            i += 1
            continue
        if char == "-" and i + 1 < len(sql) and sql[i + 1] == "-":
            masked.extend("  ")
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                masked.append(" ")
                i += 1
            continue
        if char == "/" and i + 1 < len(sql) and sql[i + 1] == "*":
            masked.extend("  ")
            i += 2
            while i < len(sql):
                if sql[i] == "*" and i + 1 < len(sql) and sql[i + 1] == "/":
                    masked.extend("  ")
                    i += 2
                    break
                masked.append(" ")
                i += 1
            continue
        if char in {"'", '"'}:
            quote_char = char
            masked.append(" ")
            i += 1
            continue
        masked.append(char)
        i += 1
    return "".join(masked)


def check_sql_quality(sql: str) -> list[str]:
    warnings: list[str] = []
    policy_sql_upper = _mask_sql_literals_identifiers_and_comments(str(sql or "")).upper()
    if re.search(r"\bORDER\s+BY\b", policy_sql_upper) and not re.search(r"\bNULLS\s+(?:LAST|FIRST)\b", policy_sql_upper):
        warnings.append("ORDER BY without NULLS LAST or NULLS FIRST is not allowed.")
    if "/" in policy_sql_upper and not re.search(r"\bNULLIF\s*\(", policy_sql_upper):
        warnings.append("Division without NULLIF is not allowed.")
    if (
        re.search(r"\bCOUNT\s*\(\s*\*\s*\)", policy_sql_upper)
        and not re.search(r"\bWHERE\b", policy_sql_upper)
        and not re.search(r"\bGROUP\s+BY\b", policy_sql_upper)
    ):
        warnings.append("COUNT(*) without WHERE or GROUP BY is not allowed.")
    return warnings


def validate_sql(sql: str) -> tuple[bool, str | None]:
    policy_sql = _mask_sql_literals_identifiers_and_comments(str(sql or ""))
    sql_upper = policy_sql.strip().upper()
    if is_unanswerable_response(sql):
        return True, None
    if not sql_upper.startswith(("SELECT", "WITH")):
        return False, "Only read-only SELECT/WITH SQL is allowed."
    if re.search(r";\s*\S", sql_upper):
        return False, "SQL contains multiple statements, which is not allowed."
    if any(re.search(rf"\b{re.escape(keyword)}\b", sql_upper) for keyword in _DANGEROUS_SQL):
        return False, "SQL contains disallowed mutating or DDL keywords."

    try:
        import sqlglot

        sqlglot.parse_one(sql, dialect="duckdb")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover
        return False, str(exc)
    quality_errors = check_sql_quality(sql)
    if quality_errors:
        return False, quality_errors[0]
    return True, None
