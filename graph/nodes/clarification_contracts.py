from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from db_rag.service.classifier import classify_clarification_reply

from ..state import MetaKeys

CLARIFICATION_KIND_DATASET_SELECTION = "generate_code_dataset_selection"
CLARIFICATION_KIND_GENERATE_CODE = "generate_code"
CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN = "rag_db_extraction_opt_in"
CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR = "db_rag_recoverable_error"
CLARIFICATION_KIND_QA_TOOL = "qa_tool"
CLARIFICATION_KIND_MEMORY_REFERENCE = "memory_reference_resolution"


@dataclass(frozen=True)
class ClarificationContract:
    kind: str
    owner: str
    expected_type: str
    allowed_reroute_intents: dict[str, str]
    reask_prompt: str
    max_attempts: int = 2
    allow_cancel: bool = False


@dataclass(frozen=True)
class ClarificationDecision:
    decision: str
    target_node: str | None = None
    normalized_value: Any = None
    intent: str | None = None
    reason: str = ""


CONTRACTS: dict[str, ClarificationContract] = {
    CLARIFICATION_KIND_DATASET_SELECTION: ClarificationContract(
        kind=CLARIFICATION_KIND_DATASET_SELECTION,
        owner="generate_code",
        expected_type="dataset_id",
        allowed_reroute_intents={"database_source": "rag_db_qa"},
        reask_prompt=(
            "Please reply with one listed dataset ID, or say 'database' to query "
            "the RePORT database."
        ),
    ),
    CLARIFICATION_KIND_GENERATE_CODE: ClarificationContract(
        kind=CLARIFICATION_KIND_GENERATE_CODE,
        owner="generate_code",
        expected_type="free_text_instruction",
        allowed_reroute_intents={"database_source": "rag_db_qa"},
        reask_prompt=(
            "Please reply with the specific analysis, transformation, or columns "
            "you want."
        ),
        allow_cancel=True,
    ),
    CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN: ClarificationContract(
        kind=CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
        owner="rag_db_qa",
        expected_type="yes_no",
        allowed_reroute_intents={
            "uploaded_dataset_analysis": "generate_code",
            "general_question": "qa",
        },
        reask_prompt=(
            "Please reply with 'yes' to proceed with DB-RAG extraction, "
            "or 'no' to skip it."
        ),
        allow_cancel=True,
    ),
    CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR: ClarificationContract(
        kind=CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR,
        owner="rag_db_qa",
        expected_type="free_text_instruction",
        allowed_reroute_intents={},
        reask_prompt=(
            "Please provide the missing DB-RAG recovery detail, or say 'cancel' "
            "to stop this workflow."
        ),
        allow_cancel=True,
    ),
    CLARIFICATION_KIND_QA_TOOL: ClarificationContract(
        kind=CLARIFICATION_KIND_QA_TOOL,
        owner="qa",
        expected_type="tool_missing_fields",
        allowed_reroute_intents={
            "database_source": "rag_db_qa",
            "uploaded_dataset_analysis": "generate_code",
        },
        reask_prompt="Please provide the requested tool field.",
        allow_cancel=True,
    ),
    CLARIFICATION_KIND_MEMORY_REFERENCE: ClarificationContract(
        kind=CLARIFICATION_KIND_MEMORY_REFERENCE,
        owner="orchestrator",
        expected_type="memory_reference",
        allowed_reroute_intents={},
        reask_prompt="Please reply with one listed task number or task ID.",
    ),
}


def contract_for_kind(kind: str | None) -> ClarificationContract | None:
    return CONTRACTS.get(str(kind or "").strip())


def build_expected(kind: str, **overrides: Any) -> dict[str, Any]:
    contract = contract_for_kind(kind)
    if contract is None:
        raise ValueError(f"Unknown clarification kind: {kind}")
    expected = {"type": contract.expected_type}
    expected.update({key: value for key, value in overrides.items() if value is not None})
    return expected


def _normalized_reply(reply: str) -> str:
    return " ".join(str(reply or "").strip().split())


def _valid_dataset_id(expected: dict[str, Any], reply: str) -> ClarificationDecision | None:
    allowed = [
        str(item).strip()
        for item in expected.get("allowed_values") or []
        if str(item).strip()
    ]
    normalized = _normalized_reply(reply)
    if normalized in allowed:
        return ClarificationDecision(
            decision="valid",
            target_node="generate_code",
            normalized_value=normalized,
            intent="uploaded_dataset_choice",
        )
    return None


def _valid_yes_no(reply: str, owner: str) -> ClarificationDecision | None:
    normalized = _normalized_reply(reply).casefold()
    yes_values = {"yes", "y", "proceed", "continue", "go ahead"}
    no_values = {"no", "n", "skip", "do not proceed", "don't proceed"}
    if normalized in yes_values:
        return ClarificationDecision(
            "valid",
            target_node=owner,
            normalized_value="yes",
            intent="yes",
        )
    if normalized in no_values:
        return ClarificationDecision(
            "valid",
            target_node=owner,
            normalized_value="no",
            intent="no",
        )
    return None


def _valid_free_text(reply: str, owner: str) -> ClarificationDecision | None:
    normalized = _normalized_reply(reply)
    if normalized:
        return ClarificationDecision(
            "valid",
            target_node=owner,
            normalized_value=normalized,
        )
    return None


def _valid_tool_fields(expected: dict[str, Any], reply: str) -> ClarificationDecision | None:
    normalized = _normalized_reply(reply)
    required = [
        str(item).strip()
        for item in expected.get("required_fields") or []
        if str(item).strip()
    ]
    if len(required) == 1 and normalized:
        return ClarificationDecision(
            "valid",
            target_node="qa",
            normalized_value={required[0]: normalized},
            intent="tool_missing_field",
        )
    return None


def _normalize_classifier_decision(
    *,
    contract: ClarificationContract,
    expected: dict[str, Any],
    raw: dict[str, Any],
) -> ClarificationDecision:
    decision = str(raw.get("decision") or "").strip().lower()
    intent = str(raw.get("intent") or "").strip()
    target_node = str(raw.get("target_node") or "").strip() or None
    normalized_value = raw.get("normalized_value")
    allowed_values = [
        str(item).strip()
        for item in expected.get("allowed_values") or []
        if str(item).strip()
    ]

    if decision == "valid" and contract.expected_type == "dataset_id":
        value = str(normalized_value or "").strip()
        if value in allowed_values:
            return ClarificationDecision("valid", contract.owner, value, intent)
        return ClarificationDecision("unclear")

    if decision == "valid" and (target_node is None or target_node == contract.owner):
        return ClarificationDecision("valid", contract.owner, normalized_value, intent)

    if decision == "reroute" and intent in contract.allowed_reroute_intents:
        mapped = contract.allowed_reroute_intents[intent]
        if target_node is None or target_node == mapped:
            return ClarificationDecision("reroute", mapped, normalized_value, intent)

    return ClarificationDecision("unclear")


def resolve_clarification_reply(
    *,
    meta: dict[str, Any],
    reply: str,
) -> ClarificationDecision:
    kind = str(meta.get(MetaKeys.CLARIFICATION_KIND) or "").strip()
    contract = contract_for_kind(kind)
    if contract is None:
        return ClarificationDecision("unclear", reason="unknown_kind")

    expected = dict(meta.get(MetaKeys.CLARIFICATION_EXPECTED) or {})
    expected_type = str(expected.get("type") or contract.expected_type)

    if expected_type == "dataset_id":
        deterministic = _valid_dataset_id(expected, reply)
        if deterministic is not None:
            return deterministic
    elif expected_type == "yes_no":
        deterministic = _valid_yes_no(reply, contract.owner)
        if deterministic is not None:
            return deterministic
    elif expected_type == "free_text_instruction" and not contract.allowed_reroute_intents:
        deterministic = _valid_free_text(reply, contract.owner)
        if deterministic is not None:
            return deterministic
    elif expected_type == "tool_missing_fields":
        deterministic = _valid_tool_fields(expected, reply)
        if deterministic is not None:
            return deterministic

    raw = classify_clarification_reply(
        clarification_kind=kind,
        clarification_return_node=contract.owner,
        pending_question=str(meta.get(MetaKeys.PENDING_QUESTION) or ""),
        reply=reply,
        expected=expected,
        allowed_reroute_intents=sorted(contract.allowed_reroute_intents),
        allowed_target_nodes=sorted(
            set(contract.allowed_reroute_intents.values()) | {contract.owner}
        ),
    )
    classifier_decision = _normalize_classifier_decision(
        contract=contract,
        expected=expected,
        raw=raw,
    )
    if classifier_decision.decision != "unclear":
        return classifier_decision
    if expected_type == "free_text_instruction" and not contract.allowed_reroute_intents:
        deterministic = _valid_free_text(reply, contract.owner)
        if deterministic is not None:
            return deterministic
    return classifier_decision
