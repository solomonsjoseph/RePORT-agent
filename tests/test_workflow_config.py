from graph import workflow_config


def test_workflow_tuning_constants_are_documented_in_one_place() -> None:
    assert workflow_config.MAX_ERROR_ITERATIONS == 5
    assert workflow_config.PLANNER_RECENT_TURNS == 3
    assert workflow_config.TOOL_ROUTER_RECENT_TURNS == 3
    assert workflow_config.QA_RECENT_TURNS == 10
    assert workflow_config.CODEGEN_RECENT_TURNS == 10
    assert workflow_config.ERROR_HANDLER_RECENT_TURNS == 10


def test_related_debug_context_limits_are_documented_in_one_place() -> None:
    assert workflow_config.CLARIFICATION_WITH_PENDING_RECENT_TURNS == 5
    assert workflow_config.CLARIFICATION_RECENT_TURNS == 3
    assert workflow_config.WORKFLOW_TRACE_TAIL == 8
    assert workflow_config.RECENT_OBSERVATIONS_LIMIT == 6
    assert workflow_config.PLANNER_DECISION_TRACE_LIMIT == 5
    assert workflow_config.PLANNER_RAW_RESPONSE_PREVIEW_CHARS == 300
