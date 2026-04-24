# DB-RAG Retrieval Enhancements Design

## Goal

Improve `db_rag` retrieval quality by aligning two retrieval-time behaviors with the sibling `RePORT-txt2sql` project:

- include numeric `min` and `max` in indexed column profile text
- enable default-on LLM query decomposition before retrieval, with fallback to the original single-query retrieval path when decomposition does not produce multiple usable concepts

The existing staged DB-RAG flow remains intact:

- retrieve context
- answer metadata questions from context
- prepare column selection for SQL-needed questions
- prepare SQL from approved selection
- execute prepared SQL

## Problem

The current `db_rag` retrieval path has two limitations compared with the sibling project:

1. indexed column profile text omits numeric ranges, which can remove useful retrieval cues for age, counts, scores, or other bounded numeric variables
2. retrieval uses the raw user question as a single query string, which can miss relevant columns for multi-concept questions such as "age, sex, diabetes, and final outcome"

The sibling project already addresses both concerns:

- chunk text includes numeric `min` and `max` when the source column is numeric
- retrieval always attempts LLM-based decomposition first, then falls back to single-query retrieval when decomposition yields only one usable concept

## Design Principles

- Match the sibling project’s decomposition logic closely, but keep the current DB-RAG architecture.
- Improve retrieval without introducing reranking in this pass.
- Preserve the current downstream DB-RAG service API.
- Keep failure behavior conservative: if decomposition fails, fall back to the original question.
- Avoid new provider configuration; decomposition uses the existing `self.llm`.

## Numeric Profiling Enhancement

`db_rag/bootstrap.py` extends `_profile_column()` so numeric source columns also include:

- `min`
- `max`

This information is appended to the same `Profile:` line already stored in column chunk text.

Example profile content:

```text
Profile: dtype: int64 | null_rate: 0/120 | samples: 18, 21, 34 | distinct_count: 42 | min: 18 | max: 79
```

This change only affects newly built Chroma indexes. Existing indexes must be rebuilt before retrieval can use the richer chunk text.

## Query Decomposition

`db_rag/service.py` adds a helper equivalent in behavior to the sibling project’s `decompose_query()`:

- prompt the LLM to extract distinct concepts from the user question
- return one short search phrase per line
- strip numbering, bullets, and commentary
- discard empty or obviously invalid lines

Example:

```text
Question: How many males with diabetes had poor final outcome

gender sex
diabetes blood sugar
final outcome treatment outcome
```

## Retrieval Flow

`DbRagService.retrieve_context(question)` changes from single-query retrieval to this flow:

1. call `decompose_query(question)`
2. if decomposition returns 0 or 1 usable phrases, run the existing single-query retrieval logic with the original question
3. otherwise, for each sub-query:
   - retrieve table hits from Chroma
   - merge unique tables by table name
4. after table merging, for each sub-query:
   - retrieve column hits from Chroma
   - filter to discovered tables
   - merge unique columns by `table.column`
5. build `DbRagContext` from the merged table and column hits

This mirrors the sibling project’s default-on decomposition and multi-query merge behavior, but stops short of adding reranking or concept reservation.

## Fallback Behavior

Fallback must be conservative:

- decomposition LLM call raises an error: fall back to single-query retrieval
- decomposition returns empty output: fall back to single-query retrieval
- decomposition returns one usable phrase: fall back to single-query retrieval

The fallback query remains the original user question, not the single returned phrase. This matches the intended behavior of preserving the previous baseline retrieval path whenever decomposition does not produce a meaningful multi-concept split.

## Scope Boundaries

Included in this pass:

- numeric `min/max` profiling
- default-on query decomposition
- multi-subquery table/column retrieval merge

Not included in this pass:

- reranking
- paired-form injection
- retrieval score debugging UI
- changes to SQL prompts
- changes to the DB-RAG approval/execution state machine

## Code Structure

`db_rag/bootstrap.py`:

- extend `_profile_column()` to add numeric `min` and `max`

`db_rag/service.py`:

- add `decompose_query(question)` helper
- add internal helpers for single-query retrieval and multi-query retrieval merge if needed for readability
- update `retrieve_context(question)` to use the decomposition-first flow

No new files are required.

## Error Handling

- numeric profiling errors should not crash indexing for non-numeric columns; numeric range extraction should only run for columns recognized as numeric
- decomposition parse failures should not block retrieval; they trigger fallback
- duplicate tables or columns returned from multiple sub-queries must be deduplicated deterministically

## Testing

Manual verification should cover:

1. rebuild DB-RAG assets and confirm numeric column chunk text now includes `min` and `max`
2. ask a single-concept query and confirm retrieval still works through the fallback path
3. ask a multi-concept query and confirm retrieval includes tables/columns contributed by multiple sub-queries
4. simulate or observe decomposition failure and confirm the service falls back to the original single-query retrieval behavior
