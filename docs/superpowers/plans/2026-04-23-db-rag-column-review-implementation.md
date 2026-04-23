# DB-RAG Column Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild DB-RAG SQL preparation so retrieval/schema-linking happens first, selected columns are reviewed by the human in an interrupt loop, SQL is generated only from approved columns, and successful SQL execution persists a thread-scoped dataset artifact.

**Architecture:** Keep `rag_db_qa` as the owner of DB-RAG retrieval, QA, column selection, SQL generation, SQL execution, and dataset registration. Add one deterministic interrupt node, `rag_db_column_review`, that only pauses for human review and records approve/revise/cancel feedback. Reuse the existing `llm` object with separate structured prompts instead of adding a separate model/provider.

**Tech Stack:** Python, LangGraph interrupts, Streamlit, ChromaDB, DuckDB, LangChain message classes, pandas/Parquet dataset artifacts, pytest.

---

## File Map

- Modify `db_rag/service.py`: add typed DB-RAG result dataclasses, public retrieval/context methods, structured QA classification, column selection preparation, SQL candidate preparation, and prepared-SQL execution.
- Modify `graph/nodes/rag_db_qa.py`: replace `pending_sql_offer` flow with metadata-QA, column-review preparation, approved-selection SQL generation, and prepared-SQL execution.
- Create `graph/nodes/rag_db_column_review.py`: DB-RAG-specific interrupt node for approve/revise/cancel of selected columns.
- Modify `graph/nodes/action_metadata.py`: include `rag_db_column_review` in action metadata loading.
- Modify `graph/nodes/node_registry.py`: add deterministic readiness for `rag_db_column_review`.
- Modify `graph/builder.py`: wire the new node into the graph.
- Modify `graph/nodes/orchestrator/policy.py`: add `rag_db_column_review` to known actions and deterministic control actions through the YAML contract.
- Modify `docs/superpowers/specs/orchestrator-gating-policy.md`: include `rag_db_column_review` in deterministic control actions.
- Modify `graph/nodes/orchestrator/action_mask.py`: add block reason for inactive DB-RAG column review.
- Modify `graph/nodes/orchestrator/workflow_status.py`: classify pending column review and pending SQL confirmation as `blocked_waiting`.
- Create `UI/ui_rag_db_column_review.py`: Streamlit UI for selected-column review.
- Modify `utils/streamlit_interrupts.py`: recognize DB-RAG column-review interrupts.
- Modify `streamlit_app.py`: render the DB-RAG column-review UI.
- Modify `utils/dataset_artifacts.py`: no storage path change; ensure provenance/marker metadata from DB-RAG execution is preserved.
- Test `tests/test_db_rag_service.py`: service method contracts and SQL validation behavior.
- Test `tests/test_rag_db_qa_node.py`: node state transitions.
- Test `tests/test_rag_db_column_review_node.py`: interrupt node feedback handling.
- Test `tests/test_orchestrator_db_routing.py`, `tests/test_orchestrator_workflow_status.py`, `tests/test_orchestrator_action_mask.py`, `tests/test_node_registry.py`, `tests/test_streamlit_interrupts.py`, `tests/test_dataset_artifacts.py`: integration contracts.

## Pre-Flight

- [ ] **Step 1: Confirm dirty worktree before implementation**

Run: `git status --short`

Expected: existing unrelated dirty files may be present. Do not revert or overwrite unrelated changes. Only edit files listed in this plan.

- [ ] **Step 2: Read the approved spec**

Run: `sed -n '1,320p' docs/superpowers/specs/2026-04-23-db-rag-qa-node-redesign.md`

Expected: spec includes `rag_db_column_review`, approved-column SQL generation, and `runtime/datasets/{thread_id}/` artifact storage.

---

### Task 1: Add DB-RAG Service Data Contracts

**Files:**
- Modify: `db_rag/service.py`
- Test: `tests/test_db_rag_service.py`

- [ ] **Step 1: Write failing tests for typed service return shapes**

Append to `tests/test_db_rag_service.py`:

```python
def test_db_rag_context_dataclasses_round_trip() -> None:
    from db_rag.service import DbRagColumnHit, DbRagContext, DbRagTableHit

    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 1A", text="Table: Form 1A")],
        columns=[
            DbRagColumnHit(
                table="Form 1A",
                column="AGE",
                text="Column: AGE\nDescription: age",
            )
        ],
        table_context="Table: Form 1A",
        column_context="Column: AGE\nDescription: age",
    )

    assert context.table_names == ["Form 1A"]
    assert context.column_names == ["AGE"]
    assert context.columns[0].as_prompt_line() == "Form 1A.AGE"
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/test_db_rag_service.py::test_db_rag_context_dataclasses_round_trip -v`

Expected: FAIL with import error for `DbRagColumnHit`, `DbRagContext`, or `DbRagTableHit`.

- [ ] **Step 3: Implement minimal dataclasses**

Add near the top of `db_rag/service.py`, after constants:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DbRagTableHit:
    table: str
    text: str


@dataclass(frozen=True)
class DbRagColumnHit:
    table: str
    column: str
    text: str

    def as_prompt_line(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
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
```

- [ ] **Step 4: Run test and verify it passes**

Run: `pytest tests/test_db_rag_service.py::test_db_rag_context_dataclasses_round_trip -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add db_rag/service.py tests/test_db_rag_service.py
git commit -m "Add DB-RAG service data contracts"
```

---

### Task 2: Make Retrieval Public and Typed

**Files:**
- Modify: `db_rag/service.py`
- Test: `tests/test_db_rag_service.py`

- [ ] **Step 1: Write failing test for `retrieve_context()`**

Append to `tests/test_db_rag_service.py`:

```python
def test_retrieve_context_returns_typed_hits(monkeypatch) -> None:
    from db_rag.service import DbRagService

    class _Collection:
        def __init__(self, result):
            self.result = result

        def query(self, **_kwargs):
            return self.result

    table_result = {
        "documents": [["Table: Form 1A"]],
        "metadatas": [[{"table": "Form 1A"}]],
    }
    column_result = {
        "documents": [["Column: AGE", "Column: SEX", "Column: OTHER"]],
        "metadatas": [[
            {"table": "Form 1A", "column": "AGE"},
            {"table": "Form 1A", "column": "SEX"},
            {"table": "Form 2A", "column": "OTHER"},
        ]],
    }

    service = DbRagService(llm=object())
    monkeypatch.setattr(
        service,
        "_load_collections",
        lambda: (_Collection(table_result), _Collection(column_result)),
    )

    context = service.retrieve_context("age and sex")

    assert context.table_names == ["Form 1A"]
    assert context.column_names == ["AGE", "SEX"]
    assert "Table: Form 1A" in context.table_context
    assert "Column: AGE" in context.column_context
    assert "Column: OTHER" not in context.column_context
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/test_db_rag_service.py::test_retrieve_context_returns_typed_hits -v`

Expected: FAIL because `DbRagService.retrieve_context` is missing.

- [ ] **Step 3: Implement `retrieve_context()` and keep `_retrieve_context()` as internal adapter only if needed**

In `DbRagService`, replace direct callers of `_retrieve_context()` with `retrieve_context()`. Implement:

```python
def retrieve_context(self, question: str) -> DbRagContext:
    table_collection, column_collection = self._load_collections()

    table_result = table_collection.query(
        query_texts=[question],
        n_results=4,
        include=["documents", "metadatas"],
    )
    tables = [
        DbRagTableHit(table=metadata["table"], text=document)
        for document, metadata in zip(table_result["documents"][0], table_result["metadatas"][0])
    ]

    selected_tables = [entry.table for entry in tables]
    column_result = column_collection.query(
        query_texts=[question],
        n_results=12,
        include=["documents", "metadatas"],
    )
    columns: list[DbRagColumnHit] = []
    for document, metadata in zip(column_result["documents"][0], column_result["metadatas"][0]):
        if metadata["table"] not in selected_tables:
            continue
        columns.append(
            DbRagColumnHit(
                table=metadata["table"],
                column=metadata["column"],
                text=document,
            )
        )

    return DbRagContext(
        tables=tables,
        columns=columns,
        table_context="\n\n".join(entry.text for entry in tables),
        column_context="\n\n".join(entry.text for entry in columns),
    )
```

- [ ] **Step 4: Run service tests**

Run: `pytest tests/test_db_rag_service.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add db_rag/service.py tests/test_db_rag_service.py
git commit -m "Expose typed DB-RAG retrieval context"
```

---

### Task 3: Add Structured QA Classification

**Files:**
- Modify: `db_rag/service.py`
- Test: `tests/test_db_rag_service.py`

- [ ] **Step 1: Write failing tests for metadata QA vs SQL-needed QA**

Append to `tests/test_db_rag_service.py`:

```python
def test_answer_from_context_returns_metadata_qa_without_sql() -> None:
    from db_rag.service import DbRagColumnHit, DbRagContext, DbRagService, DbRagTableHit

    class _LLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"answer":"This database contains TB study forms.","needs_sql":false,"rationale":"overview only"}'})()

    service = DbRagService(llm=_LLM())
    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 1A", text="Table: Form 1A")],
        columns=[DbRagColumnHit(table="Form 1A", column="AGE", text="Column: AGE")],
        table_context="Table: Form 1A",
        column_context="Column: AGE",
    )

    answer = service.answer_from_context("Give me overview of the database", context)

    assert answer.answer == "This database contains TB study forms."
    assert answer.needs_sql is False
    assert answer.relevant_tables == ["Form 1A"]
    assert answer.relevant_columns == ["AGE"]


def test_answer_from_context_marks_subset_request_as_sql_needed() -> None:
    from db_rag.service import DbRagColumnHit, DbRagContext, DbRagService, DbRagTableHit

    class _LLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"answer":"AGE and SEX appear relevant.","needs_sql":true,"rationale":"row-level subset requested"}'})()

    service = DbRagService(llm=_LLM())
    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 1A", text="Table: Form 1A")],
        columns=[DbRagColumnHit(table="Form 1A", column="AGE", text="Column: AGE")],
        table_context="Table: Form 1A",
        column_context="Column: AGE",
    )

    answer = service.answer_from_context("subset age and sex", context)

    assert answer.needs_sql is True
    assert answer.rationale == "row-level subset requested"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_db_rag_service.py::test_answer_from_context_returns_metadata_qa_without_sql tests/test_db_rag_service.py::test_answer_from_context_marks_subset_request_as_sql_needed -v`

Expected: FAIL because `answer_from_context()` and `DbRagQaAnswer` are missing or unstructured.

- [ ] **Step 3: Implement `DbRagQaAnswer` and structured JSON parsing**

Add:

```python
@dataclass(frozen=True)
class DbRagQaAnswer:
    answer: str
    needs_sql: bool
    rationale: str
    relevant_tables: list[str]
    relevant_columns: list[str]
```

Add helper:

```python
def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or "").strip())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
```

Add method:

```python
def answer_from_context(self, question: str, context: DbRagContext) -> DbRagQaAnswer:
    response = self.llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a DB-RAG QA assistant for the RePORT clinical database. "
                    "Use only the retrieved table and column context. "
                    "Return JSON with keys answer, needs_sql, rationale. "
                    "Set needs_sql=true only for row-level extraction, counts, cohort filters, "
                    "aggregations, exact numeric answers, or subset creation. "
                    "Set needs_sql=false for metadata/schema/database overview questions."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Table context:\n{context.table_context or 'none'}\n\n"
                    f"Column context:\n{context.column_context or 'none'}"
                )
            ),
        ]
    )
    payload = _parse_json_object(coerce_text_content(getattr(response, "content", "")))
    answer = str(payload.get("answer") or "").strip()
    return DbRagQaAnswer(
        answer=answer or "I could not answer from the retrieved DB-RAG context.",
        needs_sql=payload.get("needs_sql") is True,
        rationale=str(payload.get("rationale") or "").strip(),
        relevant_tables=context.table_names,
        relevant_columns=context.column_names,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_db_rag_service.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add db_rag/service.py tests/test_db_rag_service.py
git commit -m "Add structured DB-RAG QA classification"
```

---

### Task 4: Add Column Selection Candidate Preparation

**Files:**
- Modify: `db_rag/service.py`
- Test: `tests/test_db_rag_service.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_db_rag_service.py`:

```python
def test_prepare_column_selection_uses_feedback_history() -> None:
    from db_rag.service import DbRagColumnHit, DbRagContext, DbRagService, DbRagTableHit

    captured = {}

    class _LLM:
        def invoke(self, messages):
            captured["prompt"] = messages[-1].content
            return type("Response", (), {
                "content": (
                    '{"selection_id":"sel-1","rationale":"Age, sex, and outcome are needed.",'
                    '"tables":["Form 1A"],'
                    '"columns":[{"table":"Form 1A","column":"AGE","description":"age"},'
                    '{"table":"Form 1A","column":"SEX","description":"gender"}]}'
                )
            })()

    service = DbRagService(llm=_LLM())
    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 1A", text="Table: Form 1A")],
        columns=[DbRagColumnHit(table="Form 1A", column="AGE", text="Column: AGE")],
        table_context="Table: Form 1A",
        column_context="Column: AGE",
    )

    selection = service.prepare_column_selection(
        "subset age and sex",
        context,
        feedback_history=[{"feedback": "Include gender explicitly."}],
    )

    assert selection.selection_id == "sel-1"
    assert selection.tables == ["Form 1A"]
    assert selection.columns[0]["column"] == "AGE"
    assert "Include gender explicitly." in captured["prompt"]
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_db_rag_service.py::test_prepare_column_selection_uses_feedback_history -v`

Expected: FAIL because `prepare_column_selection()` and `ColumnSelectionCandidate` are missing.

- [ ] **Step 3: Implement candidate dataclass and method**

Add:

```python
@dataclass(frozen=True)
class ColumnSelectionCandidate:
    selection_id: str
    question: str
    tables: list[str]
    columns: list[dict[str, str]]
    rationale: str
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "awaiting_review"
```

Add method:

```python
def prepare_column_selection(
    self,
    question: str,
    context: DbRagContext,
    feedback_history: list[dict[str, Any]] | None = None,
) -> ColumnSelectionCandidate:
    feedback = list(feedback_history or [])
    feedback_text = json.dumps(feedback, indent=2, ensure_ascii=False)
    response = self.llm.invoke(
        [
            SystemMessage(
                content=(
                    "You select exact database tables and columns for a future SQL query. "
                    "Use only retrieved context. Return JSON with keys selection_id, rationale, tables, columns. "
                    "columns must be a list of objects with table, column, description."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Table context:\n{context.table_context or 'none'}\n\n"
                    f"Column context:\n{context.column_context or 'none'}\n\n"
                    f"Human feedback history:\n{feedback_text}"
                )
            ),
        ]
    )
    payload = _parse_json_object(coerce_text_content(getattr(response, "content", "")))
    raw_columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
    columns = [
        {
            "table": str(item.get("table") or "").strip(),
            "column": str(item.get("column") or "").strip(),
            "description": str(item.get("description") or "").strip(),
        }
        for item in raw_columns
        if isinstance(item, dict) and item.get("table") and item.get("column")
    ]
    tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
    selection_id = str(payload.get("selection_id") or f"selection-{abs(hash(question)) % 100000}").strip()
    return ColumnSelectionCandidate(
        selection_id=selection_id,
        question=question,
        tables=[str(table).strip() for table in tables if str(table).strip()],
        columns=columns,
        rationale=str(payload.get("rationale") or "").strip(),
        feedback_history=feedback,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_db_rag_service.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add db_rag/service.py tests/test_db_rag_service.py
git commit -m "Add DB-RAG column selection preparation"
```

---

### Task 5: Add SQL Candidate Preparation and Prepared Execution

**Files:**
- Modify: `db_rag/service.py`
- Test: `tests/test_db_rag_service.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_db_rag_service.py`:

```python
def test_prepare_sql_candidate_uses_approved_columns_only() -> None:
    from db_rag.service import ColumnSelectionCandidate, DbRagService

    captured = {}

    class _LLM:
        def invoke(self, messages):
            captured["prompt"] = messages[-1].content
            return type("Response", (), {"content": 'SELECT "AGE", "SEX" FROM "Form 1A"'})()

    service = DbRagService(llm=_LLM())
    selection = ColumnSelectionCandidate(
        selection_id="sel-1",
        question="subset age and sex",
        tables=["Form 1A"],
        columns=[
            {"table": "Form 1A", "column": "AGE", "description": "age"},
            {"table": "Form 1A", "column": "SEX", "description": "sex"},
        ],
        rationale="approved",
        status="approved",
    )

    candidate = service.prepare_sql_candidate("subset age and sex", selection)

    assert candidate.sql == 'SELECT "AGE", "SEX" FROM "Form 1A"'
    assert candidate.columns == selection.columns
    assert "AGE" in captured["prompt"]
    assert "not approved" not in captured["prompt"].lower()


def test_prepare_sql_candidate_rejects_unapproved_selection() -> None:
    from db_rag.service import ColumnSelectionCandidate, DbRagService

    service = DbRagService(llm=object())
    selection = ColumnSelectionCandidate(
        selection_id="sel-1",
        question="subset age",
        tables=["Form 1A"],
        columns=[{"table": "Form 1A", "column": "AGE", "description": "age"}],
        rationale="pending",
        status="awaiting_review",
    )

    try:
        service.prepare_sql_candidate("subset age", selection)
    except ValueError as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("Expected unapproved selection to be rejected")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_db_rag_service.py::test_prepare_sql_candidate_uses_approved_columns_only tests/test_db_rag_service.py::test_prepare_sql_candidate_rejects_unapproved_selection -v`

Expected: FAIL because `PreparedSqlCandidate` or `prepare_sql_candidate()` is missing or has old signature.

- [ ] **Step 3: Implement prepared SQL dataclasses and methods**

Add:

```python
@dataclass(frozen=True)
class PreparedSqlCandidate:
    question: str
    sql: str
    tables: list[str]
    columns: list[dict[str, str]]
    selection_id: str
    status: str = "prepared"


@dataclass(frozen=True)
class SqlExecutionResult:
    answer: str
    sql: str
    dataframe: Any
    source_tables: list[str]
```

Add method:

```python
def prepare_sql_candidate(
    self,
    question: str,
    approved_selection: ColumnSelectionCandidate,
) -> PreparedSqlCandidate:
    if approved_selection.status != "approved":
        raise ValueError("SQL generation requires an approved column selection.")
    approved_context = "\n".join(
        f"- {item['table']}.{item['column']}: {item.get('description', '')}"
        for item in approved_selection.columns
    )
    response = self.llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a DuckDB SQL expert for the RePORT clinical research database. "
                    "Use only the approved tables and columns listed by the user message. "
                    "Return only read-only DuckDB SQL using SELECT or WITH."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Approved tables:\n{json.dumps(approved_selection.tables, ensure_ascii=False)}\n\n"
                    f"Approved columns:\n{approved_context}"
                )
            ),
        ]
    )
    sql = _extract_sql(coerce_text_content(getattr(response, "content", "")))
    valid, error = _validate_sql(sql)
    if not valid:
        raise ValueError(error or "SQL validation failed.")
    return PreparedSqlCandidate(
        question=question,
        sql=sql,
        tables=list(approved_selection.tables),
        columns=list(approved_selection.columns),
        selection_id=approved_selection.selection_id,
    )
```

Modify execution to accept a prepared candidate:

```python
def execute_prepared_sql(self, candidate: PreparedSqlCandidate) -> SqlExecutionResult:
    import duckdb

    valid, error = _validate_sql(candidate.sql)
    if not valid:
        raise ValueError(error or "SQL validation failed.")
    db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    dataframe = db.execute(candidate.sql).fetchdf()
    return SqlExecutionResult(
        answer=f"Read-only SQL execution completed with {len(dataframe)} result row(s).",
        sql=candidate.sql,
        dataframe=dataframe,
        source_tables=list(candidate.tables),
    )
```

Keep `execute_sql_flow()` temporarily only if existing tests still call it, but make it call `retrieve_context()`, `prepare_column_selection()`, `prepare_sql_candidate()` only with an approved selection in tests. Remove it in the final cleanup task.

- [ ] **Step 4: Run service tests**

Run: `pytest tests/test_db_rag_service.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add db_rag/service.py tests/test_db_rag_service.py
git commit -m "Prepare DB-RAG SQL from approved columns"
```

---

### Task 6: Add `rag_db_column_review` Interrupt Node

**Files:**
- Create: `graph/nodes/rag_db_column_review.py`
- Test: `tests/test_rag_db_column_review_node.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rag_db_column_review_node.py`:

```python
from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _fresh_module(feedback):
    langgraph_types = ModuleType("langgraph.types")
    langgraph_types.interrupt = lambda _payload: feedback
    sys.modules["langgraph.types"] = langgraph_types
    sys.modules.pop("graph.nodes.rag_db_column_review", None)
    return importlib.import_module("graph.nodes.rag_db_column_review")


def test_rag_db_column_review_approve_marks_selection_approved() -> None:
    mod = _fresh_module({"action": "approve"})
    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "question": "subset age",
                    "selection_id": "sel-1",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                    "rationale": "age requested",
                    "feedback_history": [],
                    "status": "awaiting_review",
                }
            }
        },
        "messages": [],
    }

    updated = mod.rag_db_column_review_node(state)

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "approved"
    assert review["columns"][0]["column"] == "AGE"


def test_rag_db_column_review_revision_appends_feedback() -> None:
    mod = _fresh_module({"action": "revise", "feedback": "Use final outcome too."})
    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "question": "subset age",
                    "selection_id": "sel-1",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                    "rationale": "age requested",
                    "feedback_history": [],
                    "status": "awaiting_review",
                }
            }
        },
        "messages": [],
    }

    updated = mod.rag_db_column_review_node(state)

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "needs_revision"
    assert review["feedback_history"][-1]["feedback"] == "Use final outcome too."
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_rag_db_column_review_node.py -v`

Expected: FAIL because module is missing.

- [ ] **Step 3: Implement node**

Create `graph/nodes/rag_db_column_review.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from langgraph.types import interrupt

NODE_NAME = "rag_db_column_review"
NODE_CAPABILITY = "Ask human review of DB-RAG selected tables and columns before SQL generation."


def rag_db_column_review_node(state):
    agents = dict(state.get("agents") or {})
    rag_state = dict(agents.get("rag_db_qa") or {})
    review = dict(rag_state.get("pending_column_review") or {})
    feedback = interrupt(
        {
            "type": "rag_db_column_review",
            "question": review.get("question", ""),
            "selection_id": review.get("selection_id", ""),
            "tables": list(review.get("tables") or []),
            "columns": list(review.get("columns") or []),
            "rationale": review.get("rationale", ""),
            "feedback_history": list(review.get("feedback_history") or []),
        }
    )

    action = str(feedback.get("action") or "").strip().lower()
    history = list(review.get("feedback_history") or [])
    if action == "approve":
        review["status"] = "approved"
    elif action == "revise":
        review["status"] = "needs_revision"
        history.append(
            {
                "feedback": str(feedback.get("feedback") or "").strip(),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        review["feedback_history"] = history
    elif action == "cancel":
        review["status"] = "cancelled"
        rag_state.pop("pending_sql_candidate", None)
    else:
        review["status"] = "needs_revision"
        history.append(
            {
                "feedback": f"Unrecognized review action: {action}",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        review["feedback_history"] = history

    rag_state["pending_column_review"] = review
    agents["rag_db_qa"] = rag_state
    return {**state, "agents": agents}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag_db_column_review_node.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/rag_db_column_review.py tests/test_rag_db_column_review_node.py
git commit -m "Add DB-RAG column review interrupt node"
```

---

### Task 7: Wire `rag_db_column_review` into Graph and Orchestrator Controls

**Files:**
- Modify: `graph/nodes/action_metadata.py`
- Modify: `graph/nodes/node_registry.py`
- Modify: `graph/builder.py`
- Modify: `docs/superpowers/specs/orchestrator-gating-policy.md`
- Modify: `graph/nodes/orchestrator/policy.py`
- Modify: `graph/nodes/orchestrator/action_mask.py`
- Test: `tests/test_node_registry.py`
- Test: `tests/test_orchestrator_policy_contract.py`
- Test: `tests/test_orchestrator_action_mask.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_node_registry.py`:

```python
def test_rag_db_column_review_ready_when_selection_waits_for_review() -> None:
    import importlib

    registry = importlib.import_module("graph.nodes.node_registry")
    review = registry.NODE_REGISTRY_MAP["rag_db_column_review"]
    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "status": "awaiting_review",
                    "columns": [{"table": "Form 1A", "column": "AGE"}],
                }
            }
        }
    }

    assert review.is_ready(state) is True
```

Append to `tests/test_orchestrator_action_mask.py`:

```python
def test_action_mask_blocks_inactive_rag_db_column_review() -> None:
    from graph.nodes.orchestrator.action_mask import mask_actions

    allowed, blocked = mask_actions({"agents": {"rag_db_qa": {}}}, ["qa", "rag_db_column_review"])

    assert allowed == ["qa"]
    assert blocked["rag_db_column_review"] == "requires DB-RAG column selection awaiting review"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_node_registry.py::test_rag_db_column_review_ready_when_selection_waits_for_review tests/test_orchestrator_action_mask.py::test_action_mask_blocks_inactive_rag_db_column_review -v`

Expected: FAIL because action metadata and registry entries are missing.

- [ ] **Step 3: Wire action metadata**

Add `"rag_db_column_review"` to `_ACTION_MODULES` in `graph/nodes/action_metadata.py` immediately after `"rag_db_qa"`.

- [ ] **Step 4: Add registry predicate**

Add helper to `graph/nodes/node_registry.py`:

```python
def _has_pending_rag_db_column_review(state: AgentState) -> bool:
    rag_state = get_agent_state(state, "rag_db_qa")
    review = dict(rag_state.get("pending_column_review") or {})
    return review.get("status") == "awaiting_review"
```

Add `NodeDefinition` after `rag_db_qa`:

```python
NodeDefinition(
    name="rag_db_column_review",
    capability=ACTION_CAPABILITIES["rag_db_column_review"],
    priority=46,
    is_ready=_has_pending_rag_db_column_review,
),
```

- [ ] **Step 5: Wire graph builder**

Import and add action:

```python
from .nodes.rag_db_column_review import rag_db_column_review_node
```

In `action_nodes`:

```python
"rag_db_column_review": _run_and_mark("rag_db_column_review", rag_db_column_review_node),
```

- [ ] **Step 6: Update deterministic policy contract and known actions**

In `docs/superpowers/specs/orchestrator-gating-policy.md`, add:

```yaml
  - rag_db_column_review
```

after `tool_handler`.

In `graph/nodes/orchestrator/policy.py`, add `"rag_db_column_review"` to `known_actions`.

- [ ] **Step 7: Update action mask reason**

In `graph/nodes/orchestrator/action_mask.py`, add:

```python
"rag_db_column_review": _static_reason("requires DB-RAG column selection awaiting review"),
```

- [ ] **Step 8: Run routing/registry tests**

Run: `pytest tests/test_node_registry.py tests/test_orchestrator_policy_contract.py tests/test_orchestrator_action_mask.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add graph/nodes/action_metadata.py graph/nodes/node_registry.py graph/builder.py docs/superpowers/specs/orchestrator-gating-policy.md graph/nodes/orchestrator/policy.py graph/nodes/orchestrator/action_mask.py tests/test_node_registry.py tests/test_orchestrator_action_mask.py
git commit -m "Wire DB-RAG column review into graph"
```

---

### Task 8: Update Workflow Status for DB-RAG Waiting States

**Files:**
- Modify: `graph/nodes/orchestrator/workflow_status.py`
- Test: `tests/test_orchestrator_workflow_status.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_orchestrator_workflow_status.py`:

```python
def test_pending_rag_db_column_review_is_blocked_waiting() -> None:
    state = {
        "output": {"qa_response": "Review these columns."},
        "agents": {
            "executor": {"run_status": "idle"},
            "rag_db_qa": {
                "pending_column_review": {
                    "status": "awaiting_review",
                    "selection_id": "sel-1",
                }
            },
        },
        "meta": {"workflow_trace": ["orchestrator", "rag_db_qa"]},
        "last_action": "rag_db_qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_rag_db_column_review"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_rag_db_column_review:sel-1"


def test_pending_rag_db_sql_confirmation_is_blocked_waiting() -> None:
    state = {
        "output": {"qa_response": "Run this SQL?"},
        "agents": {
            "executor": {"run_status": "idle"},
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "selection_id": "sel-1",
                    "status": "prepared",
                }
            },
        },
        "meta": {"workflow_trace": ["orchestrator", "rag_db_qa"]},
        "last_action": "rag_db_qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_rag_db_sql_confirmation"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_rag_db_sql_confirmation:sel-1"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_orchestrator_workflow_status.py::test_pending_rag_db_column_review_is_blocked_waiting tests/test_orchestrator_workflow_status.py::test_pending_rag_db_sql_confirmation_is_blocked_waiting -v`

Expected: FAIL because statuses are not derived.

- [ ] **Step 3: Implement status derivation before generic answered detection**

In `derive_workflow_status()`, after tool-request handling and before terminal/error/code states:

```python
rag_state = get_node_data(state, "rag_db_qa") or dict((state.get("agents") or {}).get("rag_db_qa") or {})
pending_review = dict(rag_state.get("pending_column_review") or {})
if pending_review.get("status") == "awaiting_review":
    selection_id = str(pending_review.get("selection_id") or "unknown")
    return {
        "milestone": "awaiting_rag_db_column_review",
        "completion_status": "blocked_waiting",
        "blocker_signature": f"waiting_for_rag_db_column_review:{selection_id}",
    }

pending_sql = dict(rag_state.get("pending_sql_candidate") or {})
if pending_sql.get("status") == "prepared":
    selection_id = str(pending_sql.get("selection_id") or "unknown")
    return {
        "milestone": "awaiting_rag_db_sql_confirmation",
        "completion_status": "blocked_waiting",
        "blocker_signature": f"waiting_for_rag_db_sql_confirmation:{selection_id}",
    }
```

- [ ] **Step 4: Run workflow-status tests**

Run: `pytest tests/test_orchestrator_workflow_status.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/orchestrator/workflow_status.py tests/test_orchestrator_workflow_status.py
git commit -m "Classify DB-RAG review waiting states"
```

---

### Task 9: Rewrite `rag_db_qa` Node State Flow

**Files:**
- Modify: `graph/nodes/rag_db_qa.py`
- Test: `tests/test_rag_db_qa_node.py`

- [ ] **Step 1: Replace old tests that assert `pending_sql_offer`**

Update `tests/test_rag_db_qa_node.py` so no test asserts `pending_sql_offer`. Add tests:

```python
def test_rag_db_metadata_qa_answers_without_pending_sql_state() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            assert "overview" in question
            return {"context": "typed context"}

        def answer_from_context(self, question, context):
            return type("Answer", (), {
                "answer": "Overview from metadata.",
                "needs_sql": False,
                "rationale": "metadata only",
                "relevant_tables": ["Form 1A"],
                "relevant_columns": ["AGE"],
            })()

    state = {
        "messages": [_HumanMessage("Give me overview of the database")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert "Overview from metadata." in updated["output"]["qa_response"]
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]


def test_rag_db_sql_needed_request_creates_column_review_without_sql() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            return {"context": question}

        def answer_from_context(self, question, context):
            return type("Answer", (), {
                "answer": "AGE and SEX appear relevant.",
                "needs_sql": True,
                "rationale": "row-level subset",
                "relevant_tables": ["Form 1A"],
                "relevant_columns": ["AGE", "SEX"],
            })()

        def prepare_column_selection(self, question, context, feedback_history=None):
            return type("Selection", (), {
                "selection_id": "sel-1",
                "question": question,
                "tables": ["Form 1A"],
                "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                "rationale": "age requested",
                "feedback_history": [],
                "status": "awaiting_review",
            })()

        def prepare_sql_candidate(self, question, approved_selection):
            raise AssertionError("SQL must not be generated before column approval")

    state = {
        "messages": [_HumanMessage("subset age among index cases")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "awaiting_review"
    assert review["selection_id"] == "sel-1"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_rag_db_qa_node.py -v`

Expected: FAIL because old node still uses `pending_sql_offer`.

- [ ] **Step 3: Add serialization helpers in `graph/nodes/rag_db_qa.py`**

Add:

```python
def _object_to_dict(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    data = {}
    for key in (
        "selection_id",
        "question",
        "tables",
        "columns",
        "rationale",
        "feedback_history",
        "status",
        "sql",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data
```

- [ ] **Step 4: Implement new node flow**

In `rag_db_qa_node()`:

```python
rag_state = get_agent_state(state, "rag_db_qa")
latest_question = latest_user_message(state)
question = str(question_override or _question_from_stale_qa_followup(state, latest_question) or latest_question)

if provider not in _SUPPORTED_PROVIDERS:
    updated = _append_ai_response(
        state,
        "DB-RAG currently requires an OpenAI or Anthropic provider. Switch the model provider and try again.",
    )
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    return update_agent_state(updated, "rag_db_qa", {"status": "done", "active_thread": False})

readiness = service.readiness() if service is not None else {"ready": False, "message": "DB-RAG service is unavailable."}
if not readiness.get("ready"):
    updated = _append_ai_response(state, str(readiness.get("message") or "DB-RAG assets are not ready."))
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    return update_agent_state(updated, "rag_db_qa", {"status": "done", "active_thread": False})

pending_sql = dict(rag_state.get("pending_sql_candidate") or {})
if pending_sql and _is_explicit_sql_confirmation(latest_question):
    result = service.execute_prepared_sql(_dict_to_prepared_sql_candidate(pending_sql))
    return _persist_sql_result(state, rag_state, pending_sql, result)

pending_review = dict(rag_state.get("pending_column_review") or {})
if pending_review.get("status") == "needs_revision":
    context = rag_state.get("last_retrieval_context_obj") or service.retrieve_context(pending_review.get("question") or question)
    selection = service.prepare_column_selection(
        pending_review.get("question") or question,
        context,
        feedback_history=list(pending_review.get("feedback_history") or []),
    )
    selection_payload = _object_to_dict(selection)
    selection_payload["status"] = "awaiting_review"
    updated = _append_ai_response(state, _format_column_review_response(selection_payload))
    return update_agent_state(updated, "rag_db_qa", {**rag_state, "pending_column_review": selection_payload, "active_thread": True})

if pending_review.get("status") == "approved" and not pending_sql:
    selection_payload = {**pending_review, "status": "approved"}
    candidate = service.prepare_sql_candidate(question or selection_payload.get("question"), _dict_to_column_selection(selection_payload))
    candidate_payload = _object_to_dict(candidate)
    updated = _append_ai_response(state, _format_sql_candidate_response(candidate_payload))
    return update_agent_state(updated, "rag_db_qa", {**rag_state, "pending_sql_candidate": candidate_payload, "active_thread": True})

context = service.retrieve_context(question)
answer = service.answer_from_context(question, context)
if not answer.needs_sql:
    updated = _append_ai_response(state, answer.answer)
    return update_agent_state(updated, "rag_db_qa", {"status": "done", "active_thread": True, "last_database_question": question})

selection = service.prepare_column_selection(question, context, feedback_history=[])
selection_payload = _object_to_dict(selection)
selection_payload["status"] = "awaiting_review"
updated = _append_ai_response(state, _format_column_review_response(selection_payload))
return update_agent_state(updated, "rag_db_qa", {"status": "done", "active_thread": True, "last_database_question": question, "pending_column_review": selection_payload})
```

Implement `_is_explicit_sql_confirmation()` narrowly:

```python
def _is_explicit_sql_confirmation(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    return normalized in {"yes", "yes run it", "run it", "execute it", "execute the query", "run the prepared sql"}
```

- [ ] **Step 5: Add tests for approved review and execution persistence**

Add:

```python
def test_rag_db_approved_column_review_generates_sql_candidate() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def prepare_sql_candidate(self, question, approved_selection):
            assert approved_selection.status == "approved"
            return type("Candidate", (), {
                "question": question,
                "sql": 'SELECT "AGE" FROM "Form 1A"',
                "tables": ["Form 1A"],
                "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                "selection_id": "sel-1",
                "status": "prepared",
            })()

    state = {
        "messages": [_HumanMessage("continue")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "question": "subset age",
                    "selection_id": "sel-1",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                    "rationale": "approved",
                    "feedback_history": [],
                    "status": "approved",
                }
            }
        },
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert updated["agents"]["rag_db_qa"]["pending_sql_candidate"]["sql"].startswith("SELECT")
    assert "Run this read-only SQL" in updated["output"]["qa_response"]
```

- [ ] **Step 6: Run node tests**

Run: `pytest tests/test_rag_db_qa_node.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add graph/nodes/rag_db_qa.py tests/test_rag_db_qa_node.py
git commit -m "Refactor DB-RAG node around column review state"
```

---

### Task 10: Persist SQL Subsets with Marker Metadata

**Files:**
- Modify: `graph/nodes/rag_db_qa.py`
- Test: `tests/test_rag_db_qa_node.py`
- Test: `tests/test_dataset_artifacts.py`

- [ ] **Step 1: Write failing persistence test**

Append to `tests/test_rag_db_qa_node.py`:

```python
def test_rag_db_sql_execution_persists_subset_with_marker_metadata(tmp_path, monkeypatch) -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def execute_prepared_sql(self, candidate):
            return type("Result", (), {
                "answer": "Read-only SQL execution completed with 1 result row(s).",
                "sql": candidate.sql,
                "dataframe": pd.DataFrame({"AGE": [42]}),
                "source_tables": ["Form 1A"],
            })()

    monkeypatch.setattr(rag, "DEFAULT_RUNTIME_ROOT", tmp_path, raising=False)
    state = {
        "messages": [_HumanMessage("run it")],
        "output": {},
        "observations": [],
        "meta": {"thread_id": "thread-1"},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-1",
                    "feedback_history": [{"feedback": "Use age only."}],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "subset age",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "age"}],
                    "selection_id": "sel-1",
                    "status": "prepared",
                },
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    dataset_id = updated["artifacts"]["active_dataset_id"]
    artifact = updated["artifacts"]["datasets"][dataset_id]
    assert artifact["kind"] == "subset"
    assert artifact["provenance"]["source"] == "db_rag_sql"
    assert artifact["provenance"]["feedback_history"] == [{"feedback": "Use age only."}]
    assert artifact["provenance"]["selected_columns"][0]["column"] == "AGE"
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_rag_db_qa_node.py::test_rag_db_sql_execution_persists_subset_with_marker_metadata -v`

Expected: FAIL until node execution persistence includes marker metadata.

- [ ] **Step 3: Implement persistence helper**

In `graph/nodes/rag_db_qa.py`, import `DEFAULT_RUNTIME_ROOT`:

```python
from utils.dataset_artifacts import DEFAULT_RUNTIME_ROOT, persist_dataset_artifact, register_dataset_artifact
```

Add helper:

```python
def _persist_sql_result(state, rag_state: dict, candidate: dict, result) -> AgentState:
    dataset_id = f"subset-{uuid.uuid4().hex[:8]}"
    review = dict(rag_state.get("pending_column_review") or {})
    artifact = persist_dataset_artifact(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        thread_id=_thread_id_from_state(state),
        dataset_id=dataset_id,
        kind="subset",
        dataframe=result.dataframe,
        schema=None,
        provenance={
            "source": "db_rag_sql",
            "question": candidate.get("question"),
            "sql": result.sql,
            "source_tables": list(result.source_tables),
            "selected_tables": list(candidate.get("tables") or []),
            "selected_columns": list(candidate.get("columns") or []),
            "selection_id": candidate.get("selection_id"),
            "feedback_history": list(review.get("feedback_history") or []),
        },
    )
    updated = register_dataset_artifact(state, artifact, make_active=True)
    updated = _append_ai_response(updated, f"{result.answer}\n\nSQL used:\n{result.sql}\n\nSaved dataset: {dataset_id}")
    next_rag_state = dict(rag_state)
    next_rag_state.pop("pending_sql_candidate", None)
    next_rag_state.pop("pending_column_review", None)
    next_rag_state["status"] = "done"
    next_rag_state["active_thread"] = True
    return update_agent_state(updated, "rag_db_qa", next_rag_state)
```

- [ ] **Step 4: Run persistence tests**

Run: `pytest tests/test_rag_db_qa_node.py::test_rag_db_sql_execution_persists_subset_with_marker_metadata tests/test_dataset_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/rag_db_qa.py tests/test_rag_db_qa_node.py tests/test_dataset_artifacts.py
git commit -m "Persist DB-RAG SQL subsets with provenance"
```

---

### Task 11: Add Streamlit UI for Column Review Interrupt

**Files:**
- Create: `UI/ui_rag_db_column_review.py`
- Modify: `utils/streamlit_interrupts.py`
- Modify: `streamlit_app.py`
- Test: `tests/test_streamlit_interrupts.py`

- [ ] **Step 1: Write failing interrupt utility test**

Append to `tests/test_streamlit_interrupts.py`:

```python
def test_should_render_rag_db_column_review_interrupt() -> None:
    from types import SimpleNamespace
    from utils.streamlit_interrupts import should_render_review_interrupt

    interrupt_event = SimpleNamespace(
        id="int-rag",
        value={"type": "rag_db_column_review"},
    )

    assert should_render_review_interrupt(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={},
    ) is True
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_streamlit_interrupts.py::test_should_render_rag_db_column_review_interrupt -v`

Expected: FAIL because utility does not recognize `rag_db_column_review`.

- [ ] **Step 3: Update interrupt utility**

In `utils/streamlit_interrupts.py`, add:

```python
if ui_type == "rag_db_column_review":
    return True
```

before the final `return False`.

- [ ] **Step 4: Create UI module**

Create `UI/ui_rag_db_column_review.py`:

```python
import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)


def ui_rag_db_column_review(app, config, payload, interrupt_id, queue_resume):
    ui_type = "rag_db_column_review"
    st.subheader("Review DB-RAG Columns Before SQL Generation")
    st.caption("SQL will be generated only after you approve these selected tables and columns.")

    if payload.get("question"):
        st.markdown("Source question:")
        st.code(payload["question"], language="text")

    if payload.get("rationale"):
        st.markdown("Selection rationale:")
        st.markdown(payload["rationale"])

    columns = list(payload.get("columns") or [])
    if columns:
        st.markdown("Selected columns:")
        for item in columns:
            table = item.get("table", "")
            column = item.get("column", "")
            description = item.get("description", "")
            st.markdown(f"- `{table}.{column}` - {description}")

    history = list(payload.get("feedback_history") or [])
    if history:
        st.markdown("Previous feedback:")
        for entry in history:
            st.markdown(f"- {entry.get('feedback', '')}")

    feedback_key = f"rag_db_column_feedback_{interrupt_id}"
    feedback = st.text_area(
        "Feedback for regenerated column selection",
        key=feedback_key,
        height=120,
    ).strip()

    col1, col2, col3 = st.columns(3)
    approve = col1.button("Approve Columns", key=f"{ui_type}_approve_{interrupt_id}")
    revise = col2.button("Regenerate Selection", key=f"{ui_type}_revise_{interrupt_id}")
    cancel = col3.button("Cancel SQL Prep", key=f"{ui_type}_cancel_{interrupt_id}")

    if approve:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "approve"})
        st.rerun()

    if revise:
        if not feedback:
            st.error("Please enter feedback before regenerating the column selection.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "revise", "feedback": feedback})
        st.rerun()

    if cancel:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "cancel"})
        st.rerun()
```

- [ ] **Step 5: Wire Streamlit app**

In `streamlit_app.py`, import:

```python
from UI.ui_rag_db_column_review import ui_rag_db_column_review
```

In interrupt rendering:

```python
elif ui_type == "rag_db_column_review":
    ui_rag_db_column_review(app, config, payload, interrupt_id, queue_interrupt_resume)
```

- [ ] **Step 6: Run UI-related tests**

Run: `pytest tests/test_streamlit_interrupts.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add UI/ui_rag_db_column_review.py utils/streamlit_interrupts.py streamlit_app.py tests/test_streamlit_interrupts.py
git commit -m "Add Streamlit DB-RAG column review UI"
```

---

### Task 12: Orchestrator Integration Tests for DB-RAG Review Loop

**Files:**
- Modify: `tests/test_orchestrator_db_routing.py`
- Modify: `tests/test_orchestrator_prompt.py`

- [ ] **Step 1: Add orchestrator test for column-review deterministic routing**

Append to `tests/test_orchestrator_db_routing.py`:

```python
def test_orchestrator_routes_pending_rag_db_column_review_before_planner() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [],
        "output": {"qa_response": "Review columns."},
        "observations": [],
        "last_action": "rag_db_qa",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {
                "pending_column_review": {
                    "status": "awaiting_review",
                    "selection_id": "sel-1",
                }
            },
        },
        "meta": {"workflow_trace": ["orchestrator", "rag_db_qa"]},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM('{"action":"qa","thought":"wrong"}'),
        ["qa", "rag_db_qa", "rag_db_column_review", "end"],
    )

    assert updated["next_action"] == "rag_db_column_review"
```

- [ ] **Step 2: Run test and verify failure if previous integration missed something**

Run: `pytest tests/test_orchestrator_db_routing.py::test_orchestrator_routes_pending_rag_db_column_review_before_planner -v`

Expected: PASS if graph controls are already correct. If FAIL, fix deterministic action ordering or registry readiness before continuing.

- [ ] **Step 3: Run orchestrator test suite**

Run: `pytest tests/test_orchestrator_db_routing.py tests/test_orchestrator_prompt.py tests/test_orchestrator_workflow_status.py tests/test_orchestrator_action_mask.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_orchestrator_db_routing.py tests/test_orchestrator_prompt.py
git commit -m "Cover DB-RAG column review orchestration"
```

---

### Task 13: Remove Legacy `pending_sql_offer` Path and Update Tests

**Files:**
- Modify: `graph/nodes/rag_db_qa.py`
- Modify: `tests/test_rag_db_qa_node.py`
- Modify: `tests/test_clarification_node.py`
- Modify: `tests/test_orchestrator_prompt.py`

- [ ] **Step 1: Search for legacy state**

Run: `rg -n "pending_sql_offer|rag_db_sql_offer|execute_sql_flow" graph tests`

Expected: remaining hits show exactly what still needs migration.

- [ ] **Step 2: Delete legacy offer state**

Remove:

```python
_SQL_OFFER = "Do you want me to extract a read-only subset or run a read-only SQL query for this?"
```

Remove the `pending_sql_offer` branch from `rag_db_qa_node()`.

Remove the `set_clarification_meta` call that sets `kind="rag_db_sql_offer"` from DB-RAG SQL preparation. Column review is now an interrupt, not a clarification.

- [ ] **Step 3: Update clarification tests**

Replace old `rag_db_sql_offer` clarification expectations with `rag_db_column_review` interrupt expectations. If a test only verifies `clarification_node` can resume `rag_db_qa`, keep that test for stale QA follow-up takeover and remove SQL-offer-specific state.

- [ ] **Step 4: Run search again**

Run: `rg -n "pending_sql_offer|rag_db_sql_offer|execute_sql_flow" graph tests`

Expected: no hits, unless `execute_sql_flow` is intentionally kept only in `db_rag/service.py` for backward compatibility during this task. If retained, add a comment-free test proving new node flow does not call it.

- [ ] **Step 5: Run migrated tests**

Run: `pytest tests/test_rag_db_qa_node.py tests/test_clarification_node.py tests/test_orchestrator_prompt.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add graph/nodes/rag_db_qa.py tests/test_rag_db_qa_node.py tests/test_clarification_node.py tests/test_orchestrator_prompt.py
git commit -m "Remove legacy DB-RAG SQL offer flow"
```

---

### Task 14: Full Verification

**Files:**
- No code changes unless verification exposes failures.

- [ ] **Step 1: Run focused DB-RAG suite**

Run: `pytest tests/test_db_rag_service.py tests/test_rag_db_qa_node.py tests/test_rag_db_column_review_node.py tests/test_orchestrator_db_routing.py tests/test_orchestrator_workflow_status.py tests/test_streamlit_interrupts.py tests/test_dataset_artifacts.py -v`

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run: `pytest`

Expected: all tests pass. Warnings from protobuf deprecations are acceptable if they match the current baseline.

- [ ] **Step 3: Inspect diff**

Run: `git status --short`

Expected: only intentional files changed. Existing unrelated dirty files from before this plan may still appear; do not stage unrelated changes.

- [ ] **Step 4: Commit final cleanup if needed**

If Step 1 or Step 2 required small fixes:

Run `git status --short`, identify only the files changed for the verification fix, then stage those exact paths with `git add path/to/file.py path/to/test.py` and commit with:

```bash
git commit -m "Stabilize DB-RAG column review workflow"
```

If no fixes were needed, do not create an empty commit.
