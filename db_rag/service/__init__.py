from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DbRagUnanswerableError
from .intent_service import DbRagIntentMixin
from .retrieval_service import DbRagRetrievalMixin
from .selection_service import DbRagSelectionMixin
from .sql_service import DbRagSqlMixin


@dataclass
class DbRagService(
    DbRagRetrievalMixin,
    DbRagIntentMixin,
    DbRagSelectionMixin,
    DbRagSqlMixin,
):
    llm: Any
    indexing_model: str | None = None


__all__ = ["DbRagService", "DbRagUnanswerableError"]
