from __future__ import annotations

import ast
import hashlib

def is_executable_python(code: str) -> bool:
    snippet = (code or "").strip()
    if not snippet:
        return False
    try:
        ast.parse(snippet)
    except SyntaxError:
        return False
    return True

def code_fingerprint(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()