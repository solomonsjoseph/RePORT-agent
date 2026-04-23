# DB-RAG QA Node Redesign

## Goal

Redesign `rag_db_qa` so database questions follow the same core pipeline as `RePORT-txt2sql/code`:

1. retrieve/schema-link candidate tables and columns from Chroma
2. answer metadata questions from retrieved context with a QA prompt
3. pause for human review of the columns that will be used for row-level SQL requests
4. regenerate the column selection from human feedback until approved
5. generate SQL only from the approved column selection
6. validate SQL before it can be offered
7. execute only an explicitly prepared SQL candidate after user confirmation

The node must not infer execution intent from loose words such as "subset", "extract", or "columns".

## Problem

The current node mixes three different responsibilities:

- answering database metadata questions
- deciding whether a follow-up confirms SQL execution
- generating/executing SQL for the active question

This makes the state `pending_sql_offer=True` too broad. After an overview answer, a new request such as "Help me subset age, gender, diabetes status, and final outcome among index case" can be misread as confirmation to execute SQL for the previous overview question. That is a design flaw, not a parser bug.

## Design Principles

- Retrieval always comes first for DB-RAG turns.
- Metadata QA does not need SQL.
- Row-level SQL requests require human approval of selected tables/columns before SQL generation.
- SQL generation and SQL execution are separate stages.
- Confirmation applies only to a concrete prepared SQL candidate, not to a vague offer.
- `rag_db_qa` stores structured state, not natural-language interpretation state.
- The same `llm` object can be reused with separate prompts. The boundary is prompt/role and typed method, not provider configuration.

## Request Classes

`rag_db_qa` handles three request classes:

1. Metadata QA
   - Examples: database overview, table descriptions, column meanings, forms available, join keys.
   - Behavior: retrieve table/column context and call the QA prompt.
   - SQL: none.

2. SQL-needed column preparation
   - Examples: subset records, count rows, cohort filters, exact numeric answers, aggregations.
   - Behavior: retrieve/schema-link context, call the QA prompt to summarize candidate tables/columns, then pause for human review before SQL generation.
   - SQL: not generated until the human approves the selected columns.

3. Column-selection review
   - Examples: user approves selected columns, adds missing columns, removes irrelevant columns, or asks for a regenerated selection.
   - Behavior: if approved, call the SQL prompt using only the approved tables/columns; if feedback is provided, regenerate the selection and pause again.
   - SQL: generated only after approval.

4. SQL execution confirmation
   - Examples: "yes, run it", "execute this query", "run the prepared SQL".
   - Behavior: execute the stored prepared SQL candidate.
   - SQL: execute only if the candidate exists and matches the current pending candidate.

## Service API

`DbRagService` exposes explicit stages:

```python
def retrieve_context(question: str) -> DbRagContext:
    ...

def answer_from_context(question: str, context: DbRagContext) -> DbRagQaAnswer:
    ...

def prepare_column_selection(
    question: str,
    context: DbRagContext,
    feedback_history: list[dict] | None = None,
) -> ColumnSelectionCandidate:
    ...

def prepare_sql_candidate(
    question: str,
    approved_selection: ColumnSelectionCandidate,
) -> PreparedSqlCandidate:
    ...

def execute_prepared_sql(candidate: PreparedSqlCandidate) -> SqlExecutionResult:
    ...
```

The methods may all use `self.llm`, but each method has a dedicated prompt and return shape.

## Structured State

`agents["rag_db_qa"]` replaces `pending_sql_offer` with structured pending state:

```python
{
  "active_thread": true,
  "last_database_question": "...",
  "last_retrieval_context": {
    "tables": [...],
    "columns": [...]
  },
  "pending_column_review": {
    "question": "...",
    "selection_id": "...",
    "tables": [...],
    "columns": [...],
    "rationale": "...",
    "feedback_history": [
      {
        "feedback": "...",
        "created_at": "..."
      }
    ],
    "status": "awaiting_review" | "approved" | "needs_revision"
  },
  "pending_sql_candidate": {
    "question": "...",
    "sql": "...",
    "tables": [...],
    "columns": [...],
    "status": "prepared"
  }
}
```

Only `pending_column_review.status == "approved"` allows SQL generation. Only `pending_sql_candidate` creates an execution-confirmation state. A metadata answer alone does not create executable state.

## Human Review Interrupt

SQL-needed requests create a `pending_column_review` and route to a dedicated DB-RAG column-review interrupt node named `rag_db_column_review`.

The interrupt payload includes:

- source question
- selected tables
- selected exact column names
- column descriptions and allowed/sample values when available
- LLM rationale for why each column was selected
- feedback history from previous review rounds

The UI supports three actions:

- approve: accept the selected columns and continue to SQL generation
- revise: provide feedback and regenerate the column selection
- cancel: stop SQL preparation and keep the metadata answer visible

The review loop can repeat until the user approves or cancels. Feedback is appended to `feedback_history` and passed back into `prepare_column_selection()` so regeneration is grounded in the user's correction.

## Graph Integration

Add `rag_db_column_review` as a deterministic control action.

Readiness:

- `agents["rag_db_qa"]["pending_column_review"]["status"] == "awaiting_review"`

Resume behavior:

- approve: set the review status to `approved`, preserve the reviewed selection, and route back to `rag_db_qa` for SQL generation
- revise: append feedback, set the review status to `needs_revision`, and route back to `rag_db_qa` for regenerated column selection
- cancel: set the review status to `cancelled`, clear pending SQL state, and end the DB-RAG SQL-preparation workflow

`rag_db_qa` owns the retrieval, QA answer, regenerated column selection, SQL generation, and SQL execution stages. `rag_db_column_review` only pauses, captures human feedback, and records the review decision.

## Node Flow

Each `rag_db_qa` invocation follows this order:

1. Check readiness.
2. If the latest user turn is an explicit confirmation and a `pending_sql_candidate` exists, execute that candidate.
3. Otherwise treat the latest user turn as a new or follow-up DB-RAG question.
4. Retrieve/schema-link tables and columns from Chroma.
5. Classify whether the request needs SQL preparation.
6. For metadata QA, answer from retrieved context and stop.
7. For SQL-needed requests, prepare a column-selection candidate and pause for human review.
8. If the column selection is approved, prepare SQL from the approved selection, validate it, present the SQL, and ask for execution confirmation.

This means "Help me subset age..." always performs fresh retrieval for that new request, shows the columns that would be used, accepts human correction, and only then prepares SQL.

## Output Behavior

Metadata QA response includes:

- concise answer from retrieved metadata
- relevant tables when useful
- relevant exact column names when useful

Column review response includes:

- relevant tables
- relevant exact column names
- descriptions/rationale for selected columns
- instruction that SQL will be generated only after approval

SQL candidate response includes:

- approved tables
- approved exact column names
- proposed read-only SQL
- confirmation question

SQL execution response includes:

- execution row count
- SQL used
- persisted subset artifact details

## Dataset Artifact Storage

Executed SQL-derived subsets are persisted using the existing dataset artifact convention:

`runtime/datasets/<thread_id>/`

Each successful execution writes:

- `<dataset_id>.parquet`
- `<dataset_id>.schema.json`
- `<dataset_id>.metadata.json`

The metadata file includes:

- source user question
- approved table and column selection
- full feedback history from the column-review loop
- generated SQL
- source tables
- row count and column count
- created timestamp
- marker fields identifying the artifact as `kind="subset"` and `source="db_rag_sql"`

The artifact is registered in `artifacts.datasets`, and `artifacts.active_dataset_id` points to the newest executed subset for downstream analysis.

## Error Handling

- Retrieval not ready: answer with initialization instructions and end.
- SQL validation failure: surface a concise "could not prepare valid SQL" message with relevant retrieved tables/columns; do not execute.
- SQL execution failure: surface the database error, keep the failed candidate state for inspection, and do not route into generic code generation.
- Column review cancellation: clear pending column review and pending SQL candidate, then end without execution.

## Orchestrator Contract

The existing deterministic completion rule remains:

- metadata QA answer -> complete
- column selection awaiting human review -> blocked waiting
- SQL candidate prepared and confirmation requested -> blocked waiting
- SQL executed -> complete

The orchestrator does not re-enter `rag_db_qa` until the user sends a new message.

## Non-Goals

- No separate QA model/provider configuration in this pass.
- No new reranker implementation unless already available in the current DB-RAG service.
- No automatic SQL execution from a fresh row-level request.
- No compatibility layer around `pending_sql_offer`; migrate to structured pending SQL state.
- No second dataset storage location; use `runtime/datasets/<thread_id>/`.

## Testing

Add tests for:

- database overview uses retrieval QA and does not prepare SQL
- subset request performs retrieval, prepares column review, and does not generate SQL yet
- column review feedback regenerates selected columns and preserves feedback history
- approved column review generates SQL from only the approved selected tables/columns
- confirmation executes the exact pending SQL candidate
- new contentful follow-up while a candidate is pending starts a fresh retrieval instead of executing the old candidate
- invalid SQL preparation is surfaced without execution
- successful execution persists a subset artifact under `runtime/datasets/<thread_id>/` with SQL, selected columns, feedback history, and marker metadata
- orchestrator stops after metadata QA, column-review interrupt, and SQL candidate confirmation prompt
