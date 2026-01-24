import pandas as pd
import json
from pathlib import Path
def load_context(data_path, schema_path):
    """
    Load dataset and schema from files.
    Supports CSV, TSV, JSON (records).
    """

    data_path = Path(data_path)
    schema_path = Path(schema_path)

    # -------------------------
    # Load schema (JSON only)
    # -------------------------
    with open(schema_path, "r") as f:
        schema = json.load(f)

    # -------------------------
    # Load dataset
    # -------------------------
    suffix = data_path.suffix.lower()

    if suffix in [".csv"]:
        try:
            df = pd.read_csv(data_path)
        except pd.errors.ParserError:
            # robust fallback for messy CSVs
            df = pd.read_csv(data_path, sep=None, engine="python")

    elif suffix in [".tsv", ".txt"]:
        df = pd.read_csv(data_path, sep="\t")

    elif suffix in [".json"]:
        with open(data_path, "r") as f:
            data = json.load(f)

        # Handle both JSON records and dict-of-lists
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
        else:
            raise ValueError("Unsupported JSON data format")

    else:
        raise ValueError(f"Unsupported data file type: {suffix}")

    return df, schema

def build_context(df, schema):
    """
    Build readable context for the LLM using your nested schema structure:
    {
        "column": {
            "description": "...",
            "dataType": "...",
            "notes": "..."
        },
        ...
    }
    """
    if df is None or schema is None:
        return "No dataset or schema provided."

    cols = df.columns.tolist()

    # Column list
    col_section = "\n".join(f"- {c}" for c in cols)

    # Schema descriptions
    schema_lines = []
    for col, meta in schema.items():
        description = meta.get("description", "N/A")
        dtype = meta.get("dataType", "N/A")
        notes = meta.get("notes", "")
        schema_lines.append(
            f"{col}:\n"
            f"  • Description: {description}\n"
            f"  • Type: {dtype}\n"
            f"  • Notes: {notes}\n"
        )

    schema_text = "\n".join(schema_lines)

    return (
        "Available columns:\n"
        f"{col_section}\n\n"
        "Column metadata:\n"
        f"{schema_text}"
    )
