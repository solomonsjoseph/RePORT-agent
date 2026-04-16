from pathlib import Path


def test_streamlit_uses_directory_picker_dropdown() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'working_directory_input = st.sidebar.text_input(' not in source
    assert 'st.sidebar.button(f"Open {child_dir.name}"' not in source
    assert 'selected_child_name = st.sidebar.selectbox(' in source
    assert '"Folders"' in source
    assert 'st.sidebar.button("Use selected folder")' in source


def test_streamlit_directory_picker_still_blocks_before_model_load() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    gate_index = source.index('st.sidebar.header("Workspace")')
    load_index = source.index("def load_llm(")

    assert gate_index < load_index
    assert "Choose a working directory before initializing the model." in source[gate_index:load_index]
    assert "st.stop()" in source[gate_index:load_index]
