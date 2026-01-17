def extract_python_code(text: str) -> str:
    """Extract python code from ```python``` fences."""
    if "```" not in text:
        return text.strip()

    parts = text.split("```")
    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        if lines[0].lower().startswith("python"):
            return "\n".join(lines[1:]).strip()

    # Fallback — first fenced block
    if len(parts) >= 3:
        return parts[1].strip()

    return text.strip()
