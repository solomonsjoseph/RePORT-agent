from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_streamlit_app_stubs(monkeypatch) -> None:
    sys.path.insert(0, str(REPO_ROOT))

    langchain_core = ModuleType("langchain_core")
    messages = ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content: str = "", additional_kwargs: dict | None = None) -> None:
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class HumanMessage(BaseMessage):
        type = "human"

    class AIMessage(BaseMessage):
        type = "ai"

    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    langchain_core.messages = messages

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)

    langgraph_types = ModuleType("langgraph.types")

    class Command:
        def __init__(self, resume=None) -> None:
            self.resume = resume

    langgraph_types.Command = Command
    monkeypatch.setitem(sys.modules, "langgraph.types", langgraph_types)

    graph_builder = ModuleType("graph.builder")

    class FakeApp:
        def __init__(self) -> None:
            self.snapshot = SimpleNamespace(values={}, interrupts=[], next=[])

        def get_state(self, config):
            return self.snapshot

        def invoke(self, payload, config=None):
            return {}

        def update_state(self, config, payload) -> None:
            return None

    graph_builder.build_graph = lambda llm, provider, db_path=None: FakeApp()
    monkeypatch.setitem(sys.modules, "graph.builder", graph_builder)

    llm_vllm = ModuleType("llm_vllm")
    llm_vllm.build_llm = lambda **kwargs: "fake-llm"
    llm_vllm.detect_vllm_model = lambda base_url: "fake/model"
    monkeypatch.setitem(sys.modules, "llm_vllm", llm_vllm)

    load_openai = ModuleType("UI.load_openai")
    load_openai.load_openai = lambda **kwargs: ("key", "gpt-test")
    monkeypatch.setitem(sys.modules, "UI.load_openai", load_openai)

    load_anthropic = ModuleType("UI.load_anthropic")
    load_anthropic.load_anthropic = lambda **kwargs: ("key", "claude-test")
    monkeypatch.setitem(sys.modules, "UI.load_anthropic", load_anthropic)

    import db_rag.config
    import utils.openai_models
    import utils.run_manager

    monkeypatch.setattr(db_rag.config, "resolve_db_rag_embedding_model", lambda: "embed-test")
    monkeypatch.setattr(db_rag.config, "resolve_db_rag_reranker_model", lambda: "rerank-test")
    monkeypatch.setattr(utils.openai_models, "list_supported_openai_chat_models", lambda _key: ["gpt-test"])

    class FakeRunManager:
        def __init__(self) -> None:
            self.phase = "idle"
            self.status_calls = 0

        def status(self, thread_id: str) -> dict[str, int | str]:
            self.status_calls += 1
            if self.phase == "submitted" and self.status_calls <= 2:
                return {"state": "running", "steps": 0}
            if self.phase == "submitted":
                self.phase = "idle"
            return {"state": "idle", "steps": 0}

        def is_running(self, thread_id: str) -> bool:
            return self.phase == "submitted" and self.status_calls <= 2

        def submit(self, **kwargs) -> bool:
            self.phase = "submitted"
            self.status_calls = 0
            return True

    monkeypatch.setattr(utils.run_manager, "GraphRunManager", FakeRunManager)


def test_first_submit_does_not_duplicate_conversation_blocks(monkeypatch) -> None:
    _install_streamlit_app_stubs(monkeypatch)

    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py"), default_timeout=10)
    app.run()

    assert [item.value for item in app.subheader] == ["💬 Conversation"]
    assert [button.label for button in app.button].count("🔄 Reset Conversation") == 1

    app.text_input(key="question_input").set_value("Query my database, help me subset age")
    for button in app.button:
        if button.label == "Send":
            button.click()
            break
    app.run(timeout=10)

    assert [item.value for item in app.subheader] == ["💬 Conversation"]
    assert [button.label for button in app.button].count("🔄 Reset Conversation") == 1
