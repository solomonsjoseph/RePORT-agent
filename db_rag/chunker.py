from __future__ import annotations

import json

import pandas as pd

from .config import PRIMARY_JOIN_KEY_ALIAS, SCHEMA_DIR, SECONDARY_JOIN_KEY_ALIAS
from .data_loader import profile_column


def build_table_chunks(excel_data: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table_name = schema["form_name"]
        variables = schema.get("variables", {})
        summary = schema.get("summary", "")
        prefix = schema_file.name.split(" ")[0] + "_"
        df = excel_data.get(prefix, pd.DataFrame())
        subjid_col = next((c for c in df.columns if "SUBJID" in c.upper()), None)
        fid_col = next((c for c in df.columns if c.upper().endswith("FID")), None)
        join_labels: list[str] = []
        if subjid_col:
            join_labels.append(f"{PRIMARY_JOIN_KEY_ALIAS} ({subjid_col})")
        if fid_col:
            join_labels.append(f"{SECONDARY_JOIN_KEY_ALIAS} ({fid_col})")
        join_keys = " and ".join(join_labels) or "None"
        var_lines = [f"- {name}: {info.get('description', '(no description)')}" for name, info in variables.items()]
        parts = [
            f"Table: {table_name}",
            f"Summary: {summary}",
            f"Total variables: {len(variables)}",
            f"Join key: {join_keys}",
        ]
        if not df.empty:
            parts.append(f"Row count: {len(df)}")
        parts.append("Columns:\n" + "\n".join(var_lines))
        chunks.append(
            {
                "id": f"{table_name}.summary",
                "text": "\n".join(parts),
                "metadata": {
                    "table": table_name,
                    "subjid_col": subjid_col or "",
                    "fid_col": fid_col or "",
                    "has_subjid_join": subjid_col is not None,
                    "has_fid_join": fid_col is not None,
                },
            }
        )
    return chunks


def build_column_chunks(excel_data: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table_name = schema["form_name"]
        variables = schema.get("variables", {})
        prefix = schema_file.name.split(" ")[0] + "_"
        df = excel_data.get(prefix, pd.DataFrame())
        for column, info in variables.items():
            depends_on = info.get("depends_on")
            condition = info.get("condition")
            section_context = info.get("section_context")
            parts = [
                f"Table: {table_name}",
                f"Column: {column}",
                f"Description: {info.get('description') or '(no description)'}",
            ]
            values = info.get("values")
            if values:
                parts.append("Allowed values: " + ", ".join(f"{k}={v}" for k, v in values.items()))
            if depends_on:
                parts.append(f"Depends on: {depends_on}")
            if condition:
                parts.append(f"Condition: {condition}")
            if section_context:
                parts.append(f"Section Context: {section_context}")
            if not df.empty and column in df.columns:
                profile = profile_column(df, column)
                parts.append("Profile: " + " | ".join(f"{k}: {v}" for k, v in profile.items()))
            chunks.append(
                {
                    "id": f"{table_name}.{column}",
                    "text": "\n".join(parts),
                    "metadata": {
                        "table": table_name,
                        "column": column,
                        "form_file": schema_file.name,
                        "has_dependency": depends_on is not None,
                        "has_values": values is not None,
                        "depends_on": depends_on or "",
                        "condition": condition or "",
                        "section_context": section_context or "",
                    },
                }
            )
    return chunks
