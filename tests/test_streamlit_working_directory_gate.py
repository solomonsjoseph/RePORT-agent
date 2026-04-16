from pathlib import Path

from utils.working_directory import build_execution_profile


def test_working_directory_gate_requires_selection_before_model_load() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    gate_index = source.index('working_directory_input = st.sidebar.text_input(')
    load_index = source.index("def load_llm(")

    assert gate_index < load_index
    assert "st.stop()" in source[gate_index:load_index]


def test_working_directory_gate_exports_profile_to_environment(tmp_path: Path) -> None:
    profile = build_execution_profile(tmp_path)

    assert profile["working_directory"] == str(tmp_path.resolve())
    assert profile["environment_mode"] == "project_default"
