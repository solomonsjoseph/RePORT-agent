from pathlib import Path


def test_streamlit_uses_directory_browser_instead_of_manual_text_input() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'working_directory_input = st.sidebar.text_input(' not in source
    assert '"Create directory if missing"' not in source
    assert "browser_current_directory" in source
    assert "Use this directory" in source


def test_streamlit_directory_browser_still_blocks_before_model_load() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    gate_index = source.index('st.sidebar.header("Workspace")')
    load_index = source.index("def load_llm(")

    assert gate_index < load_index
    assert "Choose a working directory before initializing the model." in source[gate_index:load_index]
    assert "st.stop()" in source[gate_index:load_index]
