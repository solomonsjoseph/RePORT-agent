from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

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


def _clarification_state(state: dict, question: str, messages) -> dict:
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
        pending_question=_latest_human_content(messages),
    )
    updated_state = {
        **state,
        "messages": msgs,
        "output": output,
        "meta": meta,
    }
    return update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": "",
            "notes": ["Model returned clarification instead of executable code."],
        },
    )


def generate_code_node(state, llm, context):
    generate_state = get_agent_state(state, "generate_code")
    messages = state.get("messages", [])
    tool_results = list(generate_state.get("tool_results", []))

    if not any(isinstance(m, HumanMessage) for m in messages):
        return state

    output = dict(state.get("output") or {})
    if tool_results:
        output["tool_results"] = format_tool_results(tool_results)

    windowed = window_messages(messages, max_turns=CODEGEN_RECENT_TURNS)
    prompt = make_generate_code_prompt().invoke(
        {
            "messages": windowed,
            "context": context,
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
        return _clarification_state(state, _FALLBACK_CLARIFICATION_QUESTION, messages)

    if payload["response_type"] == "clarification":
        return _clarification_state(state, payload["question"], messages)

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

    meta = clear_clarification_meta(state.get("meta", {}))
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
    return update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": code,
        },
    )
