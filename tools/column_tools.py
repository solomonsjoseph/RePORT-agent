def describe_column(df, schema, column: str):
    meta = schema.get(column, {})
    out = {
        "column": column,
        "description": meta.get("description", "N/A"),
        "type": meta.get("dataType", "N/A"),
        "notes": meta.get("notes", "N/A"),
        "head": df[column].head().tolist(),
        "unique_values": df[column].unique().tolist(),
    }
    return out

def list_columns(df, schema):
    return [
        {
            "column": col,
            "type": schema.get(col, {}).get("dataType", "Unknown"),
            "description": schema.get(col, {}).get("description", ""),
        }
        for col in df.columns
    ]
