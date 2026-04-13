from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_prompt_stubs() -> None:
    class _MessagesPlaceholder:
        def __init__(self, variable_name: str, optional: bool = False) -> None:
            self.variable_name = variable_name
            self.optional = optional

    class _PromptTemplate:
        def __init__(self, messages):
            self.messages = messages

        def format_prompt(self, **kwargs):
            rendered = []
            for item in self.messages:
                if isinstance(item, tuple):
                    role, template = item
                    rendered.append({"role": role, "content": template.format(**kwargs)})
                elif isinstance(item, _MessagesPlaceholder):
                    value = kwargs.get(item.variable_name, [])
                    rendered.extend(value if isinstance(value, list) else [])
            return type("_FormattedPrompt", (), {"to_messages": lambda self: rendered})()

    class _ChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return _PromptTemplate(messages)

    class _FewShotChatMessagePromptTemplate:
        def __init__(self, example_prompt=None, examples=None):
            self.example_prompt = example_prompt
            self.examples = examples or []

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
    prompts_mod.FewShotChatMessagePromptTemplate = _FewShotChatMessagePromptTemplate

    examples_mod = ModuleType("prompts.prompt_examples")
    examples_mod.FEW_SHOT_EXAMPLES = []

    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["prompts.prompt_examples"] = examples_mod


def test_generate_code_prompt_requires_json_code_or_clarification_contract() -> None:
    _install_prompt_stubs()
    sys.modules.pop("prompts.generate_prompt", None)
    prompt_mod = importlib.import_module("prompts.generate_prompt")

    assert '"response_type": "code_result"' in prompt_mod.SYSTEM_TEXT
    assert '"response_type": "clarification"' in prompt_mod.SYSTEM_TEXT
    assert "Return only valid JSON" in prompt_mod.SYSTEM_TEXT


def test_generate_code_prompt_formats_without_treating_response_type_as_variable() -> None:
    _install_prompt_stubs()
    sys.modules.pop("prompts.generate_prompt", None)
    prompt_mod = importlib.import_module("prompts.generate_prompt")

    prompt = prompt_mod.make_generate_code_prompt()
    rendered = prompt.format_prompt(
        context="ctx",
        output="out",
        tool_results="tools",
        messages=[],
    ).to_messages()

    system_text = rendered[0]["content"]
    assert '{"response_type": "code_result"' in system_text
    assert '{"response_type": "clarification"' in system_text
