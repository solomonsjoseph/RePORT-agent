import streamlit as st
import pandas as pd
import json
# from pathlib import Path
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph.builder import build_graph
from llm_vllm import build_llm
from utils.streamlit_help import REVIEW_CONFIG, derive_ui_mode, reset_execution_state

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(
    page_title="RePORT Code Agent",
    layout="wide",
)


st.title("RePORT Code Agent (LangGraph)")
st.write("Upload your **dataset CSV** and **schema JSON**, then start chatting.")

# ============================================================
# Sidebar UI — Model Configuration
# ============================================================

st.sidebar.header("⚙️ Model Settings")

default_model = 'meta-llama/Llama-3.1-8B-Instruct'
default_temp = 0.1
default_api_key = ""
# default_max_token = 1024
# Define allowed model options
model_options = [
#    "Qwen/Qwen2.5-32B",
    default_model,   # keep your default as an option too
]

model_name = st.sidebar.selectbox(
    "Model name",
    model_options,
    index=model_options.index(default_model) if default_model in model_options else 0,
    help="Choose the model you want to use."
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=default_temp,
    step=0.05,
    help="Higher temperature = more creative code."
)

api_key = st.sidebar.text_input(
    "API Key (optional)",
    value=default_api_key,
    type="password",
    help="Leave blank if using environment variable."
)

# max_tokens = st.sidebar.number_input(
#     "Max Tokens",
#     value=4096,
#     min_value=512,
#     max_value=32768,
#     step=512,
#     help="Set a token limit or leave as default."
# )

top_p = st.sidebar.number_input(
    "Top probablity",
    value=0.9,
    min_value=0.5,
    max_value=1.0,
    step=0.05,
    help="Set the top-p value, lowering it increases creativity."
)


# ============================================================
# 1. File Upload UI
# ============================================================

uploaded_csv = st.file_uploader("Upload your dataset (.csv)", type=["csv", 'json'])
uploaded_schema = st.file_uploader("Upload your schema (.json)", type=["json"])

# ============================================================
# 2. Validate Uploaded Files
# ============================================================

if uploaded_csv and uploaded_schema:
    try:
        df = pd.read_csv(uploaded_csv)

        schema_json = uploaded_schema.read().decode("utf-8")
        schema = json.loads(schema_json)

        st.success("Dataset and schema loaded successfully!")

        # Show preview
        with st.expander("📄 Preview Dataset (df.head())", expanded=False):
            st.dataframe(df.head())

        with st.expander("📚 Schema (JSON)", expanded=False):
            st.json(schema)

    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

else:
    st.info("👆 Upload both dataset.csv and schema.json to continue.")
    st.stop()

# ============================================================
# Load LLM + Graph (cached)
# ============================================================

# Caching resources so LLM + LangGraph are not recreated every turn
@st.cache_resource
def load_llm(model_name, temperature, top_p, api_key):
    # If user supplies an API key, override OPENAI_API_KEY
    if api_key:
        import os
        os.environ["OPENAI_API_KEY"] = api_key

    return build_llm(model_name = model_name, temperature = temperature, 
                     top_p = top_p, api_key = api_key,
                     base_url= 'http://localhost:8000/v1')


@st.cache_resource
def load_app(llm, df, schema):
    return build_graph(llm, df, schema, 
                       db_path='/projects/f_wj183_1/work/xutao/2025_epi_LLM/RePORTAI_db/agent_memory.db')


llm = load_llm(model_name, temperature, top_p, api_key)
app = load_app(llm, df, schema)


# ============================================================
# 3. Session state
# ============================================================

if "thread_id" not in st.session_state:
    # Use hashes so each dataset has its own memory namespace
    st.session_state.thread_id = uuid.uuid4().hex

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset conversation (keeps uploaded files)
if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.chat_history = []
    st.session_state.thread_id = uuid.uuid4().hex
    st.rerun()
# ============================================================
# 4. Chat Input
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

user_text = st.chat_input("Ask a question about your dataset!")

if user_text:
    snapshot = app.get_state(config)
    state = snapshot.values or {}
    # Reset execution-only fields BEFORE new turn
    new_state = {
        **state,
        "generated_code": None,
        "output": None,
        "error": None,
        "review_stage": None,
        "review_reason": None,
        "human_decision": None,
        "run_status": "idle"
    }

    app.update_state(config, new_state)

    app.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    
    st.rerun()


# --------------------------------------------------
# Run LangGraph (one step)
# --------------------------------------------------
snapshot = app.get_state(config)
state = snapshot.values if snapshot else None
ui_mode = derive_ui_mode(state)
st.write("UI mode is:", ui_mode)

st.write("current state values from langraph are:", state)
st.write("Next nodes:", snapshot.next)
if ui_mode == "running":
    st.write("UI state before running:", state)
    st.write("DEBUG next nodes:", snapshot.next)
    app.invoke({}, config=config) # Resume running graph
    st.rerun()

# ============================================================
# Human review UI (CONFIG-DRIVEN)
# ============================================================
def render_review(app, config, state: dict):
    stage = state["review_stage"]
    config_ui = REVIEW_CONFIG.get(stage)

    if not config_ui:
        st.error(f"Unknown review stage: {stage}")
        return
    
    st.subheader(config_ui["title"])
    # Show generated code
    if config_ui.get("show_code"):
        st.code(state.get("generated_code", ""), language="python")
    # Show output if applicable (final review)
    if config_ui.get("show_output") and state.get("output"):
        st.markdown("**Output:**")
        st.write(state["output"])

    suggestion = st.text_area(
        "Optional suggestion / edit instruction",
        key=f"suggestion_{stage}",
        height=120,
    )

    col1, col2 = st.columns(2)

    # Approve
    if col1.button(config_ui["approve_label"], key=f"approve_{stage}"):
        action = config_ui["on_approve"]

        if action == "run":
            new_state = {
                **state,
                "review_stage": None,
                "review_reason": None,
                "human_decision": "approve",
                "run_status": "pending"
            }       

        elif action == "finish":
            new_state = {
                **state,
                "review_stage": None,
                "review_reason": None,
                "human_decision": "approve",
                "run_status": "ok"
            }  
        app.update_state(config, new_state)

        st.rerun()

    # Edit / Regenerate
    if col2.button(config_ui["edit_label"], key=f"edit_{stage}"):

        messages = list(state.get("messages", []))
        content = f"User suggestion: {suggestion}" if suggestion else "User requests code edit / regeneration."
        
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content = content))
        new_state = {
                **state,
                "messages": messages,
                "review_stage": None,
                "review_reason": None,
                "human_decision": "regenerate"
            }  
        app.update_state(config, new_state)
        st.rerun()

if ui_mode == "needs_review":
    render_review(app, config, state)
    st.stop()

# ============================================================
# Final output
# ============================================================


if ui_mode == "done":
    st.session_state.chat_history = state["messages"]
    st.success("Finished")

    if state.get("output"):
        st.write(state["output"])

    st.code(state.get("generated_code", ""), language="python")

# ============================================================
# Chat history
# ============================================================
st.subheader("💬 Conversation")
for msg in state.get("messages", []):
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

