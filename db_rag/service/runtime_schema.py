from __future__ import annotations

from functools import lru_cache

from utils.performance import timing_stage

from ..config import DUCKDB_PATH


def _quote_duckdb_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


@lru_cache(maxsize=8192)
def _duckdb_runtime_column_exists_cached(db_path: str, table: str, column: str) -> bool:
    path = str(db_path or "").strip()
    if not path:
        return True

    try:
        import duckdb
    except ModuleNotFoundError:
        return True

    quoted_table = _quote_duckdb_identifier(table)
    quoted_column = _quote_duckdb_identifier(column)
    with timing_stage("db_rag.selection.runtime_column_check", table=table, column=column):
        db = duckdb.connect(path, read_only=True)
        try:
            db.execute(f"DESCRIBE SELECT {quoted_column} FROM {quoted_table} LIMIT 0")
        except Exception:
            return False
        finally:
            db.close()
    return True


def duckdb_runtime_column_exists(table: str, column: str) -> bool:
    table_name = str(table or "").strip()
    column_name = str(column or "").strip()
    if not table_name or not column_name:
        return False
    if not DUCKDB_PATH.exists():
        return True
    return _duckdb_runtime_column_exists_cached(str(DUCKDB_PATH), table_name, column_name)
