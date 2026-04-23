# DB-RAG QA Node Redesign

## Goal

Redesign `rag_db_qa` so database questions follow the same core pipeline as `RePORT-txt2sql/code`:

1. retrieve/schema-link candidate tables and columns from Chroma
2. answer metadata questions from retrieved context with a QA prompt
3. generate SQL only for row-level or computed-data requests
4. validate SQL before it can be offered
5. execute only an explicitly prepared SQL candidate after user confirmation

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

2. SQL candidate preparation
   - Examples: subset records, count rows, cohort filters, exact numeric answers, aggregations.
   - Behavior: retrieve/schema-link context, call the QA prompt to summarize candidate tables/columns, call the SQL prompt to prepare read-only SQL, validate SQL, and ask for confirmation.
   - SQL: generated and displayed, not executed.

3. SQL execution confirmation
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

def prepare_sql_candidate(question: str, context: DbRagContext) -> PreparedSqlCandidate:
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
  "pending_sql_candidate": {
    "question": "...",
    "sql": "...",
    "tables": [...],
    "columns": [...],
    "status": "prepared"
  }
}
```

Only `pending_sql_candidate` creates an execution-confirmation state. A metadata answer alone does not create executable state.

## Node Flow

Each `rag_db_qa` invocation follows this order:

1. Check readiness.
2. If the latest user turn is an explicit confirmation and a `pending_sql_candidate` exists, execute that candidate.
3. Otherwise treat the latest user turn as a new or follow-up DB-RAG question.
4. Retrieve/schema-link tables and columns from Chroma.
5. Classify whether the request needs SQL preparation.
6. For metadata QA, answer from retrieved context and stop.
7. For SQL-needed requests, prepare SQL from retrieved context, validate it, present candidate columns/tables plus SQL, and ask for confirmation.

This means "Help me subset age..." always performs fresh retrieval for that new request before SQL is prepared.

## Output Behavior

Metadata QA response includes:

- concise answer from retrieved metadata
- relevant tables when useful
- relevant exact column names when useful

SQL candidate response includes:

- relevant tables
- relevant exact column names
- proposed read-only SQL
- confirmation question

SQL execution response includes:

- execution row count
- SQL used
- persisted subset artifact details

## Error Handling

- Retrieval not ready: answer with initialization instructions and end.
- SQL validation failure: surface a concise "could not prepare valid SQL" message with relevant retrieved tables/columns; do not execute.
- SQL execution failure: surface the database error, keep the failed candidate state for inspection, and do not route into generic code generation.

## Orchestrator Contract

The existing deterministic completion rule remains:

- metadata QA answer -> complete
- SQL candidate prepared and confirmation requested -> blocked waiting
- SQL executed -> complete

The orchestrator does not re-enter `rag_db_qa` until the user sends a new message.

## Non-Goals

- No separate QA model/provider configuration in this pass.
- No new reranker implementation unless already available in the current DB-RAG service.
- No automatic SQL execution from a fresh row-level request.
- No compatibility layer around `pending_sql_offer`; migrate to structured pending SQL state.

## Testing

Add tests for:

- database overview uses retrieval QA and does not prepare SQL
- subset request performs retrieval, prepares SQL, and does not execute
- confirmation executes the exact pending SQL candidate
- new contentful follow-up while a candidate is pending starts a fresh retrieval instead of executing the old candidate
- invalid SQL preparation is surfaced without execution
- orchestrator stops after metadata QA and after SQL candidate confirmation prompt
