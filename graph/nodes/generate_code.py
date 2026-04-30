from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from ..conversation_events import (
    append_conversation_event,
    build_assistant_event,
    build_clarification_event,
    build_code_event,
    store_thread_artifact,
)
from .orchestrator.state_logic import _user_message_hash
from prompts.generate_prompt import make_generate_code_prompt
from utils.llm_response import coerce_text_content
from utils.message_window import window_messages

from .code_guardrails import code_fingerprint, is_executable_python
from .state_helpers import (
    clear_clarification_meta,
    get_agent_state,
    set_clarification_meta,
    update_agent_state,
)
from .tool_routing import format_tool_results
from ..state import MetaKeys
from ..workflow_config import CODEGEN_RECENT_TURNS
from utils.dataset_artifacts import build_dataset_context, choose_analysis_dataset

NODE_NAME = "generate_code"
NODE_CAPABILITY = (
    "Generate Python code only for explicit code-writing requests or dataset/programmatic "
    "tasks such as analysis on the user's data, plotting, transformation, or computation "
    "that should be performed in code. Do not use for general Q&A, web search, weather, "
    "or factual lookup."
)

_FALLBACK_CLARIFICATION_QUESTION = (
    "Please restate your request with the specific analysis, transformation, or columns "
    "you want in the generated code."
)


def _current_user_turn_hash(state: dict) -> str | None:
    meta = dict(state.get("meta") or {})
    return str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or _user_message_hash(state) or "") or None


def _latest_human_content(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(getattr(message, "content", "") or "").strip()
    return ""


def _default_code_summary() -> str:
    return "Prepared Python code for the requested analysis."


def _repair_generate_payload(llm, previous_response: str):
    repair_prompt = (
        "Your previous answer did not satisfy the required generate-code JSON contract.\n"
        "Return only valid JSON using exactly one of these shapes:\n"
        '{"response_type":"code_result","summary":"...","assumptions":"...","code":"..."}\n'
        '{"response_type":"clarification","question":"..."}\n'
        "Do not return markdown. Do not return a fenced code block. Do not return prose outside the JSON object.\n\n"
        f"Previous answer:\n{previous_response}"
    )
    return llm.invoke(repair_prompt)


def _parse_generate_payload(text: str) -> tuple[dict | None, str | None]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None, "empty_response"

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if not isinstance(payload, dict):
        return None, "non_object_json"

    response_type = str(payload.get("response_type", "") or "").strip()
    if response_type == "code_result":
        code = str(payload.get("code", "") or "").strip()
        summary = str(payload.get("summary", "") or "").strip() or _default_code_summary()
        assumptions = str(payload.get("assumptions", "") or "").strip()
        if not code or not is_executable_python(code):
            return None, "invalid_code"
        return {
            "response_type": "code_result",
            "summary": summary,
            "assumptions": assumptions,
            "code": code,
        }, None

    if response_type == "clarification":
        question = str(payload.get("question", "") or "").strip()
        if not question:
            return None, "missing_question"
        return {
            "response_type": "clarification",
            "question": question,
        }, None

    return None, "invalid_response_type"


def _clarification_state(
    state: dict,
    question: str,
    messages,
    *,
    pending_question: str | None,
) -> dict:
    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=question))
    output = dict(state.get("output") or {})
    output["generated_code"] = ""
    output["qa_response"] = question
    output["code_summary"] = ""
    output["code_assumptions"] = ""
    meta = set_clarification_meta(
        state.get("meta", {}),
        return_node="generate_code",
        kind="generate_code",
        pending_question=pending_question,
    )
    updated_state = {
        **state,
        "messages": msgs,
        "output": output,
        "meta": meta,
    }
    updated_state = update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": "",
            "notes": ["Model returned clarification instead of executable code."],
        },
    )
    return append_conversation_event(
        updated_state,
        build_clarification_event(
            actor="generate_code",
            user_turn_hash=_current_user_turn_hash(state),
            text=question,
            status="active",
        ),
    )


def generate_code_node(state, llm, context, question_override=None):
    generate_state = get_agent_state(state, "generate_code")
    messages = state.get("messages", [])
    tool_results = list(generate_state.get("tool_results", []))

    if not any(isinstance(m, HumanMessage) for m in messages):
        return state

    output = dict(state.get("output") or {})
    if tool_results:
        output["tool_results"] = format_tool_results(tool_results)

    resolved_context = context(state) if callable(context) else context
    latest_human = str(question_override or _latest_human_content(messages) or "").strip()
    meta = dict(state.get("meta", {}))
    if isinstance(resolved_context, dict) and resolved_context.get("runtime_datasets"):
        selected_artifact, selection_reason = choose_analysis_dataset(
            state,
            latest_user_message=latest_human,
        )
        if selection_reason == "ambiguous":
            return _clarification_state(
                state,
                "Which dataset should I analyze: the uploaded dataset or the latest extracted subset?",
                messages,
                pending_question=latest_human,
            )
        if selected_artifact is not None:
            resolved_context = build_dataset_context(selected_artifact)
            meta[MetaKeys.ANALYSIS_DATASET_ID] = selected_artifact["id"]
        else:
            resolved_context = "No dataset or schema provided."
    windowed = window_messages(messages, max_turns=CODEGEN_RECENT_TURNS)
    prompt = make_generate_code_prompt().invoke(
        {
            "messages": windowed,
            "context": resolved_context,
            "output": output,
            "tool_results": output.get("tool_results", ""),
        }
    )
    response = llm.invoke(prompt)
    response_text = coerce_text_content(response.content)
    payload, _error = _parse_generate_payload(response_text)

    if payload is None:
        repair_response = _repair_generate_payload(llm, response_text)
        response_text = coerce_text_content(repair_response.content)
        payload, _error = _parse_generate_payload(response_text)

    if payload is None:
        return _clarification_state(
            state,
            _FALLBACK_CLARIFICATION_QUESTION,
            messages,
            pending_question=latest_human,
        )

    if payload["response_type"] == "clarification":
        return _clarification_state(
            state,
            payload["question"],
            messages,
            pending_question=latest_human,
        )

    code = payload["code"]
    output = dict(state.get("output") or {})
    output.pop("error", None)
    output.pop("text", None)
    output.pop("figure_png", None)
    output["generated_code"] = code
    output["qa_response"] = payload["summary"]
    output["code_summary"] = payload["summary"]
    output["code_assumptions"] = payload["assumptions"]

    human_review_state = dict(state.get("agents", {}).get("human_review", {}))
    human_review_state["before_run_decision"] = None
    human_review_state["final_decision"] = None
    human_review_state["approved_code_hash"] = None

    agents = dict(state.get("agents", {}))
    agents["human_review"] = human_review_state
    executor_state = dict(agents.get("executor", {}))
    if executor_state:
        executor_state["status"] = "idle"
        executor_state["run_status"] = "idle"
        executor_state.pop("error", None)
        agents["executor"] = executor_state

    meta = clear_clarification_meta(meta)
    meta[MetaKeys.ERROR_ITERATIONS] = 0
    meta[MetaKeys.CURRENT_CODE_HASH] = code_fingerprint(code)
    meta.pop(MetaKeys.FINAL_APPROVED_CODE_HASH, None)
    meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)

    updated_state = {
        **state,
        "output": output,
        "agents": agents,
        "meta": meta,
    }
    updated_state = store_thread_artifact(
        updated_state,
        {
            "kind": "code",
            "producer": "generate_code",
            "mime": "text/x-python",
            "summary": payload["summary"],
            "content": code,
        },
    )
    artifact_id = next(reversed(dict(updated_state.get("artifacts") or {}).get("files") or {}))
    updated_state = append_conversation_event(
        updated_state,
        build_assistant_event(
            actor="generate_code",
            user_turn_hash=_current_user_turn_hash(state),
            text=payload["summary"],
            status="done",
        ),
    )
    assistant_event_id = str((updated_state.get("artifacts") or {}).get("conversation_events", [])[-1]["event_id"])
    updated_state = append_conversation_event(
        updated_state,
        build_code_event(
            actor="generate_code",
            user_turn_hash=_current_user_turn_hash(state),
            artifact_id=artifact_id,
            text=payload["summary"],
            status="done",
            parent_event_id=assistant_event_id,
        ),
    )
    return update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": code,
        },
    )
