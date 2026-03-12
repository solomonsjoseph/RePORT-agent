from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


# ---------------------------------------------------------------------------
# Minimal stubs so utils.message_window can be imported without real langchain
# ---------------------------------------------------------------------------

class _HumanMessage:
    type = "human"
    def __init__(self, content: str = ""):
        self.content = content


class _AIMessage:
    type = "ai"
    def __init__(self, content: str = ""):
        self.content = content


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod


def _fresh_window_messages():
    _install_stubs()
    sys.modules.pop("utils.message_window", None)
    return importlib.import_module("utils.message_window").window_messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_list_returns_empty() -> None:
    window_messages = _fresh_window_messages()
    assert window_messages([], max_turns=5) == []


def test_fewer_turns_than_max_returns_all() -> None:
    window_messages = _fresh_window_messages()
    msgs = [_HumanMessage("q1"), _AIMessage("a1"), _HumanMessage("q2"), _AIMessage("a2")]
    result = window_messages(msgs, max_turns=10)
    assert result == msgs


def test_more_turns_than_max_returns_only_last_n() -> None:
    window_messages = _fresh_window_messages()
    # Build 5 turns
    msgs = []
    for i in range(5):
        msgs.append(_HumanMessage(f"q{i}"))
        msgs.append(_AIMessage(f"a{i}"))

    result = window_messages(msgs, max_turns=3)
    # Should contain only the last 3 turns = last 6 messages
    assert len(result) == 6
    assert result[0].content == "q2"
    assert result[-1].content == "a4"


def test_partial_turn_at_end_is_included() -> None:
    window_messages = _fresh_window_messages()
    msgs = [
        _HumanMessage("q1"), _AIMessage("a1"),
        _HumanMessage("q2"),  # no AI reply yet
    ]
    result = window_messages(msgs, max_turns=10)
    assert len(result) == 3
    assert result[-1].content == "q2"


def test_partial_turn_respects_max_turns() -> None:
    window_messages = _fresh_window_messages()
    # 3 complete turns + 1 partial
    msgs = []
    for i in range(3):
        msgs.append(_HumanMessage(f"q{i}"))
        msgs.append(_AIMessage(f"a{i}"))
    msgs.append(_HumanMessage("q3"))  # partial

    # max_turns=2 → keep last 2 turns = turn[1] + turn[2] (partial)
    result = window_messages(msgs, max_turns=2)
    assert result[0].content == "q2"
    assert result[-1].content == "q3"


def test_single_turn_returns_that_turn() -> None:
    window_messages = _fresh_window_messages()
    msgs = [_HumanMessage("hello"), _AIMessage("hi")]
    result = window_messages(msgs, max_turns=1)
    assert len(result) == 2
    assert result[0].content == "hello"
    assert result[1].content == "hi"
