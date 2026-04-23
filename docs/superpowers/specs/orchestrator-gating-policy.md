# Orchestrator Gating Policy

This document is the runtime-canonical contract for orchestrator fallback and recurrence policy.
Human-readable explanation lives here, and the YAML block below is parsed by the app at startup.

```yaml
deterministic_control_actions:
  - tool_handler
  - error_handler
  - terminal_execution_error
  - human_review_after_error
  - human_review_before_run
  - execute_code
  - human_review_before_output
planner_fallback_rules:
  - predicate: explicit_code_request
    action: generate_code
  - predicate: prefer_rag_db_qa
    action: rag_db_qa
  - predicate: always
    action: qa
recurrence:
  max_stagnation_steps: 4
  max_weak_progress_steps: 4
```
