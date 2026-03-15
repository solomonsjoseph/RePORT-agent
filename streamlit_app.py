import streamlit as st
import pandas as pd
import json
import os
import hashlib
import io
import requests
# from pathlib import Path
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph.builder import build_graph
from llm_vllm import build_llm, detect_vllm_model
from UI.ui_before_run_review import ui_before_run_review
from UI.ui_after_error_review import ui_after_error_review
from UI.ui_final_review import ui_final_review
from UI.load_openai import load_openai
# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(
    page_title="Multi Agent",
    layout="wide",
)


st.title("Multi Agent (LangGraph)")

# ============================================================
# Sidebar UI — Model Configuration
# ============================================================

st.sidebar.header("⚙️ Model Settings")
base_url = "http://localhost:8000/v1"

# default_model = 'meta-llama/Llama-3.1-8B-Instruct'
default_temp = 0.1
default_api_key = ""
default_openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
default_anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
openai_env_key = os.getenv("OPENAI_API_KEY", "")
anthropic_env_key = os.getenv("ANTHROPIC_API_KEY", "")

provider = st.sidebar.selectbox(
    "Provider",
    ["openai", "vllm"],
    index=0,
    help="Choose the model provider to use."
)

api_key = ""
model_name = ""

@st.cache_data(show_spinner=False)
def load_openai_models(effective_api_key):
    headers = {"Authorization": f"Bearer {effective_api_key}"}
    resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    model_ids = sorted({item.get("id") for item in data if item.get("id")})
    return model_ids

@st.cache_data(show_spinner=False)
def load_anthropic_models(effective_api_key):
    headers = {
        "x-api-key": effective_api_key,
        "anthropic-version": "2023-06-01",
    }
    resp = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    model_ids = sorted({item.get("id") for item in data if item.get("id")})
    return model_ids

if provider == "vllm":
    try:
        model_name = detect_vllm_model(base_url)
    except Exception as e:
        st.sidebar.error("Invalid vllm backend.")
        st.stop()
    short = model_name.split("/")[-1]
    st.sidebar.markdown("**🧠 Model:**")
    st.sidebar.markdown(f"### {short}")
    st.sidebar.caption(model_name)
elif provider == "openai":
    api_key, model_name = load_openai(
        default_api_key=default_api_key,
        openai_env_key=openai_env_key,
        default_openai_model=default_openai_model,
        load_openai_models_fn=load_openai_models,
    )

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=default_temp,
    step=0.05,
    help="Higher temperature = more creative code."
)

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
st.write("Upload your **dataset CSV** and **schema JSON**")

uploaded_csv = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
uploaded_schema = st.file_uploader("Upload your schema (.json)", type=["json"])

# ============================================================
# 2. Validate Uploaded Files
# ============================================================
dataset_signature = "no-data"
if uploaded_csv and uploaded_schema:
    try:
        
        csv_bytes = uploaded_csv.getvalue()
        schema_bytes = uploaded_schema.getvalue()

        df = pd.read_csv(io.BytesIO(csv_bytes))
        schema = json.loads(schema_bytes.decode("utf-8"))
        dataset_signature = hashlib.sha256(csv_bytes + schema_bytes).hexdigest()

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
    # st.warning("No dataset/schema provided. Running in metadata-only mode.")
    df = pd.DataFrame()
    schema = {}

# ============================================================
# Load LLM + Graph (cached)
# ============================================================

# Caching resources so LLM + LangGraph are not recreated every turn
@st.cache_resource
def load_llm(model_name, temperature, top_p, base_url, api_key, provider):
    return build_llm(model_name = model_name, temperature = temperature, 
                     top_p = top_p, base_url= base_url, api_key = api_key,
                     provider = provider)

@st.cache_resource
def load_app(llm, df, schema, dataset_signature):
    return build_graph(llm, df, schema, 
                       db_path='/projects/f_wj183_1/reflib/report-agent_db/agent_memory.db')


llm = load_llm(model_name, temperature, top_p, base_url, api_key, provider)
app = load_app(llm, df, schema, dataset_signature)

with st.sidebar.expander("🐛 Debug: LLM instance", expanded=False):
    info = {"llm_type": type(llm).__name__}

    # Common LangChain wrappers
    for attr in ["model", "model_name", "model_id"]:
        if hasattr(llm, attr):
            info[attr] = getattr(llm, attr)

    # Some wrappers store it in .client or .kwargs
    if hasattr(llm, "model_kwargs"):
        info["model_kwargs"] = getattr(llm, "model_kwargs")

    st.write(info)
# ============================================================
# 3. Session state
# ============================================================

if "thread_id" not in st.session_state:
    # Use hashes so each dataset has its own memory namespace
    st.session_state.thread_id = uuid.uuid4().hex
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! Ask me anything ...")]

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = dataset_signature
elif st.session_state.dataset_signature != dataset_signature:
    st.session_state.dataset_signature = dataset_signature
    st.info("Detected new dataset/schema. Previous conversation is preserved. Use 'Reset Conversation' to clear history.")
    st.rerun()
        
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

def initial_graph_state(user_message: HumanMessage) -> dict:
    """Bootstrap state for the first turn of a new thread.

    Later turns should send only message deltas so checkpointed graph state is preserved.
    """
    return {
        "messages": [user_message],
        "output": {},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {
                "before_run_decision": None,
                "after_error_decision": None,
                "final_decision": None,
            },
            "qa": {},
            "generate_code": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

snapshot = app.get_state(config)
has_graph_state = bool(snapshot and snapshot.values)

st.subheader("💬 Conversation")

# for msg in st.session_state.chat_history:
#     with st.chat_message(
#         "user" if isinstance(msg, HumanMessage) else "assistant"
#     ):
#         st.markdown(msg.content)
user_text = st.chat_input("Ask a question about your dataset!")
if user_text:
    user_message = HumanMessage(content=user_text)
    st.session_state.chat_history.append(user_message)

    if has_graph_state:
        # Existing thread: submit only the new user message delta so checkpointed
        # graph state (intent, clarifications, tool queue, trace) is preserved.
        app.invoke({"messages": [user_message]}, config=config)
    else:
        # New thread: seed the graph with required top-level keys once.
        app.invoke(initial_graph_state(user_message), config=config)

    st.rerun()


# --------------------------------------------------
# Run LangGraph (one step)
# --------------------------------------------------
snapshot = app.get_state(config)
state = snapshot.values if snapshot else {}

# For check workflow state, DEBUG ONLY
with st.expander("🧭 Current workflow state", expanded=False):
    executor_state = state.get("agents", {}).get("executor", {})
    review_state = state.get("agents", {}).get("human_review", {})
    st.write(
        {
            "last_action": state.get("last_action"),
            "next_action": state.get("next_action"),
            "next_nodes": list(snapshot.next or []) if snapshot else [],
            "executor_run_status": executor_state.get("run_status"),
            "before_run_decision": review_state.get("before_run_decision"),
            "after_error_decision": review_state.get("after_error_decision"),
            "final_decision": review_state.get("final_decision"),
            "workflow_trace_tail": list(state.get("meta", {}).get("workflow_trace", []))[-12:],
        }
    )

if state and state.get("messages"):
    st.session_state.chat_history = state["messages"]

# Render all previous chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

        if isinstance(msg, AIMessage):
            fig = (msg.additional_kwargs or {}).get("figure_png")
            if fig:
                st.image(fig)
                st.download_button(
                    label="⬇️ Download plot (PNG)",
                    data=fig,
                    file_name="plot.png",
                    mime="image/png",
                    key=f"dl_{id(msg)}",
                )
interrupt_event = snapshot.interrupts[0] if snapshot and snapshot.interrupts else None

# For DEBUGGING purpose, DO NOT delete, comment out in demo
####
st.write("current state values from langraph are:", state)
st.write("Next nodes:", snapshot.next)
st.write("interrupts:", snapshot.interrupts)
if interrupt_event:
    st.write("Interrupt event is:", interrupt_event)
####

if interrupt_event:
    interrupt_id = interrupt_event.id
    payload = interrupt_event.value
    ui_type = payload["type"]

    # --------------------------------------------------------
    # Review BEFORE execution
    # --------------------------------------------------------
    if ui_type == "before_run_review":
        ui_before_run_review(app, config, payload, interrupt_id)
    elif ui_type == "after_error_review":
        ui_after_error_review(app, config, payload, interrupt_id)
    elif ui_type == "final_review":
        ui_final_review(app, config, payload, interrupt_id)
    st.stop()

if snapshot and snapshot.next:
    print("Invoking next nodes:", snapshot.next)
    app.invoke({}, config=config)
    st.rerun()

# ============================================================
# Final Output
# ============================================================

output = state.get("output", {}) if state else {}
executor_ok = state.get("agents", {}).get("executor", {}).get("run_status") == "ok" if state else False
final_approved = state.get("agents", {}).get("human_review", {}).get("final_decision") == "approve" if state else False
qa_ready = bool(output.get("qa_response"))
analysis_ready = executor_ok and final_approved

# QA answers should be surfaced immediately (no final human approval required).
if qa_ready and not analysis_ready:
    st.success("Response ready")
    st.write(output.get("qa_response"))

if analysis_ready:
    st.success("Analysis completed")

    if output.get("text"):
        st.write("Output")
        st.code(output["text"], language="python")

    if output.get("generated_code"):
        st.write("Code")
        st.code(output["generated_code"], language="python")
    if output.get("figure_png"):
        st.write("Image")
        st.image(output["figure_png"])
        st.download_button(
        label="⬇️ Download plot (PNG)",
        data=output["figure_png"],
        file_name="plot.png",
        mime="image/png",
        )

# st.write("DEBUG chat types:", [type(m) for m in st.session_state.chat_history])
