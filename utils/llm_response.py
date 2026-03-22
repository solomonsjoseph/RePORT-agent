from __future__ import annotations


def coerce_text_content(content) -> str:
    """Normalize provider-specific LangChain response payloads into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
                nested = item.get("content")
                normalized = coerce_text_content(nested)
                if normalized:
                    parts.append(normalized)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            nested = getattr(item, "content", None)
            normalized = coerce_text_content(nested)
            if normalized:
                parts.append(normalized)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text:
            return text
        return coerce_text_content(content.get("content"))

    text = getattr(content, "text", None)
    if isinstance(text, str) and text:
        return text

    nested = getattr(content, "content", None)
    if nested is not None and nested is not content:
        return coerce_text_content(nested)

    return str(content)
