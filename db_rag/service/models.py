from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DbRagTableHit:
    table: str
    text: str


@dataclass
class DbRagColumnHit:
    table: str
    column: str
    text: str
    score: float | None = None

    def as_prompt_line(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class DbRagContext:
    tables: list[DbRagTableHit] = field(default_factory=list)
    columns: list[DbRagColumnHit] = field(default_factory=list)
    table_context: str = ""
    column_context: str = ""

    @property
    def table_names(self) -> list[str]:
        return [entry.table for entry in self.tables]

    @property
    def column_names(self) -> list[str]:
        return [entry.column for entry in self.columns]


@dataclass
class DbRagQaAnswer:
    answer: str
    needs_sql: bool
    rationale: str
    relevant_tables: list[str]
    relevant_columns: list[str]


@dataclass
class DbRagIntent:
    intent_id: str
    source_question: str
    goal_text: str
    mode: str
    population: str | None
    required_columns: list[Any] = field(default_factory=list)
    filters: list[Any] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    excluded_tables: list[str] = field(default_factory=list)
    excluded_columns: list[Any] = field(default_factory=list)
    feedback_history: list[dict[str, str]] = field(default_factory=list)
    status: str = "active"


@dataclass(frozen=True)
class FeedbackConstraintSet:
    goal_text: str
    required_tables: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    excluded_tables: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()


@dataclass
class ColumnSelectionCandidate:
    selection_id: str
    question: str
    tables: list[str]
    columns: list[dict[str, str]]
    rationale: str
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "awaiting_review"
    selection_source: str = "legacy"
    fallback_reason: str = ""
    raw_model_output: str = ""


@dataclass
class PreparedSqlCandidate:
    question: str
    sql: str
    tables: list[str]
    columns: list[dict[str, str]]
    selection_id: str
    status: str = "prepared"


@dataclass
class SqlExecutionResult:
    answer: str
    sql: str
    dataframe: Any
    source_tables: list[str]
