from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_rag.generation import extract_sql


def test_extract_sql_keeps_cte_root_statement() -> None:
    text = "WITH vars AS (SELECT 1 AS x) SELECT x FROM vars"
    assert extract_sql(text) == text


def test_extract_sql_keeps_cte_when_text_contains_inner_select() -> None:
    text = (
        "WITH scoped AS ("
        "SELECT column_name FROM variable_catalog WHERE column_name IN ('CBC_HGAPROP', 'CBC_HGAPCT')"
        ") SELECT column_name FROM scoped"
    )
    extracted = extract_sql(text)
    assert extracted.startswith("WITH scoped AS (")
    assert ") SELECT column_name FROM scoped" in extracted


def test_extract_sql_uses_first_sql_keyword_when_prefixed_with_text() -> None:
    text = "Here is the query:\nSELECT 1 AS value"
    assert extract_sql(text) == "SELECT 1 AS value"
