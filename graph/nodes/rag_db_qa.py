from __future__ import annotations

from typing import Any
import uuid

from langchain_core.messages import AIMessage

from ..state import AgentState
from .state_helpers import (
    clear_clarification_meta,
    get_agent_state,
    set_clarification_meta,
    update_agent_state,
)
from .tool_routing import latest_user_message
from utils.dataset_artifacts import persist_dataset_artifact, register_dataset_artifact

NODE_NAME = "rag_db_qa"
NODE_CAPABILITY = (
    "Handle database-grounded questions using retrieval over the local RePORT DB-RAG assets. "
    "Answer from retrieved schema/database context first, then optionally offer a read-only SQL "
    "extraction or subset workflow when the user confirms."
)

_SQL_OFFER = "Do you want me to extract a read-only subset or run a read-only SQL query for this?"
_SUPPORTED_PROVIDERS = {"openai", "anthropic"}
_AFFIRMATIVE = ("yes", "y", "sure", "ok", "okay", "run it", "extract", "subset")
_NEGATIVE = ("no", "n", "not now", "no thanks", "don't")


def _thread_id_from_state(state: AgentState) -> str:
    meta = dict(state.get("meta") or {})
    return str(meta.get("thread_id") or "default-thread")


def _is_affirmative(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    return any(token in normalized for token in _AFFIRMATIVE)


def _is_negative(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    return any(token in normalized for token in _NEGATIVE)


def _append_ai_response(state: AgentState, text: str) -> AgentState:
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=text))
    output = dict(state.get("output") or {})
    output["qa_response"] = text
    observations = list(state.get("observations", []))
    observations.append("rag_db_qa: responded to database question")
    return {
        **state,
        "messages": messages,
        "output": output,
        "observations": observations,
    }


def rag_db_qa_node(
    state: AgentState,
    llm,
    *,
    provider: str,
    service,
) -> AgentState:
    rag_state = get_agent_state(state, "rag_db_qa")
    question = latest_user_message(state)

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

    if rag_state.get("pending_sql_offer"):
        last_question = str(rag_state.get("last_database_question") or question)
        if _is_affirmative(question):
            sql_result = service.execute_sql_flow(last_question)
            dataset_id = f"subset-{uuid.uuid4().hex[:8]}"
            artifact = persist_dataset_artifact(
                runtime_root=None,
                thread_id=_thread_id_from_state(state),
                dataset_id=dataset_id,
                kind="subset",
                dataframe=sql_result["dataframe"],
                schema=None,
                provenance={
                    "question": last_question,
                    "sql": sql_result["sql"],
                    "source_tables": list(sql_result.get("source_tables") or []),
                },
            )
            updated = register_dataset_artifact(state, artifact, make_active=True)
            updated = _append_ai_response(updated, f'{sql_result["answer"]}\n\nSQL used:\n{sql_result["sql"]}')
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            return update_agent_state(
                updated,
                "rag_db_qa",
                {
                    "status": "done",
                    "pending_sql_offer": False,
                    "active_thread": True,
                    "last_database_question": last_question,
                },
            )

        if _is_negative(question):
            updated = _append_ai_response(state, "Okay. I will not run SQL or extract a subset unless you ask.")
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            return update_agent_state(
                updated,
                "rag_db_qa",
                {
                    "status": "done",
                    "pending_sql_offer": False,
                    "active_thread": True,
                    "last_database_question": last_question,
                },
            )

    answer_payload = service.answer_question(question)
    response_text = f'{answer_payload["answer"]}\n\n{_SQL_OFFER}'
    updated = _append_ai_response(state, response_text)
    updated["meta"] = set_clarification_meta(
        updated.get("meta") or {},
        return_node="rag_db_qa",
        kind="rag_db_sql_offer",
        pending_question=question,
    )
    return update_agent_state(
        updated,
        "rag_db_qa",
        {
            "status": "done",
            "pending_sql_offer": True,
            "active_thread": True,
            "last_database_question": question,
            "retrieval_summary": dict(answer_payload.get("retrieval_summary") or {}),
        },
    )
