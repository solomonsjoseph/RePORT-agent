from __future__ import annotations

from ..state import AgentState, MetaKeys
from .generate_code import generate_code_node
from .qa import qa_node
from .rag_db_qa import rag_db_qa_node
from .state_helpers import clear_clarification_meta
from .tool_routing import latest_user_message

NODE_NAME = "clarification"
NODE_CAPABILITY = (
    "Resume an active clarification loop by interpreting the user's follow-up and handing "
    "control back to the relevant subworkflow such as QA tool routing or code generation."
)


def _with_semantic_last_action(state: AgentState, action: str) -> AgentState:
    meta = dict(state.get("meta") or {})
    meta[MetaKeys.SEMANTIC_LAST_ACTION] = action
    return {
        **state,
        "meta": meta,
    }


def _resume_qa_tool_clarification(state: AgentState, llm) -> AgentState:
    question = latest_user_message(state)
    pending_question = (state.get("meta") or {}).get(MetaKeys.PENDING_QUESTION)
    if pending_question:
        effective_question = f"{pending_question}\n\nUser clarification: {question}"
    else:
        effective_question = question

    resumed_state = {
        **state,
        "meta": clear_clarification_meta(state.get("meta") or {}),
    }
    return _with_semantic_last_action(
        qa_node(resumed_state, llm, question_override=effective_question),
        "qa",
    )


def clarification_node(state: AgentState, llm, context: str = "") -> AgentState:
    meta = dict(state.get("meta") or {})
    kind = str(meta.get(MetaKeys.CLARIFICATION_KIND) or "")

    if kind == "qa_tool":
        return _resume_qa_tool_clarification(state, llm)

    question = latest_user_message(state)
    pending_question = meta.get(MetaKeys.PENDING_QUESTION)
    if kind in {"rag_db_extraction_opt_in", "generate_code_dataset_selection"}:
        effective_question = question
    else:
        effective_question = (
            f"{pending_question}\n\nUser clarification: {question}"
            if pending_question and question
            else question
        )
    resumed_state = {
        **state,
        "meta": clear_clarification_meta(meta),
    }

    if meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "generate_code":
        if isinstance(context, dict):
            return _with_semantic_last_action(
                generate_code_node(
                    resumed_state,
                    llm,
                    context,
                    question_override=effective_question,
                ),
                "generate_code",
            )
        return _with_semantic_last_action(
            generate_code_node(
                resumed_state,
                llm,
                context,
                question_override=effective_question,
            ),
            "generate_code",
        )
    if meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "rag_db_qa":
        if isinstance(context, dict):
            return _with_semantic_last_action(
                rag_db_qa_node(
                    resumed_state,
                    llm,
                    provider=str(context.get("provider") or ""),
                    service=context.get("db_rag_service"),
                    reranker_model=context.get("db_rag_reranker_model"),
                    question_override=effective_question,
                ),
                "rag_db_qa",
            )
        raise ValueError("rag_db_qa clarification resume requires a context mapping with provider and db_rag_service")

    return _with_semantic_last_action(
        qa_node(resumed_state, llm, context, question_override=effective_question),
        "qa",
    )
