import streamlit as st
import pandas as pd
import json
# from pathlib import Path
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from graph.builder import build_graph
from llm_vllm import build_llm, detect_vllm_model
from UI.ui_before_run_review import ui_before_run_review
from UI.ui_after_error_review import ui_after_error_review
from UI.ui_final_review import ui_final_review
# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(
    page_title="Code Agent",
    layout="wide",
)


st.title("Code Agent (LangGraph)")
st.write("Upload your **dataset CSV** and **schema JSON**, then start chatting.")


# ============================================================
# Sidebar UI — Model Configuration
# ============================================================

st.sidebar.header("⚙️ Model Settings")
base_url = "http://localhost:8000/v1"
model_name = detect_vllm_model(base_url)
short = model_name.split("/")[-1]
st.sidebar.markdown(f"**🧠 Model:**")
st.sidebar.markdown(f"### {short}")
st.sidebar.caption(model_name)

# default_model = 'meta-llama/Llama-3.1-8B-Instruct'
default_temp = 0.1
default_api_key = ""
# default_max_token = 1024
# Define allowed model options
# model_options = [
# #    "Qwen/Qwen2.5-32B",
#     default_model,   # keep your default as an option too
# ]

# model_name = st.sidebar.selectbox(
#     "Model name",
#     model_options,
#     index=model_options.index(default_model) if default_model in model_options else 0,
#     help="Choose the model you want to use."
# )

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

uploaded_csv = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
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
    st.warning("No dataset/schema provided. Running in metadata-only mode.")
    df = pd.DataFrame()
    schema = {}

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
                     base_url= base_url)


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
    st.session_state.chat_history = [AIMessage(content="Hello! Ask me anything about your dataset.")]
    
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
st.subheader("💬 Conversation")

# for msg in st.session_state.chat_history:
#     with st.chat_message(
#         "user" if isinstance(msg, HumanMessage) else "assistant"
#     ):
#         st.markdown(msg.content)
user_text = st.chat_input("Ask a question about your dataset!")
if user_text:
    messages = st.session_state.get("chat_history", [])
    st.session_state.chat_history.append(
        HumanMessage(content=user_text)
    )

    # Reset execution artifacts for new question
    new_state = {
        "output": {},
        "messages": st.session_state.chat_history,
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {
            "tool_results": [],
        },
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {
                "before_run_decision": None,
                "after_error_decision": None,
                "final_decision": None,
            },
        },
        "meta": {"error_iterations": 0},
    }

    app.invoke(new_state, config=config)
    
    st.rerun()


# --------------------------------------------------
# Run LangGraph (one step)
# --------------------------------------------------
snapshot = app.get_state(config)
state = snapshot.values if snapshot else {}
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
interrupt_event = snapshot.interrupts[0] if snapshot.interrupts else None

# For DEBUGGING purpose, DO NOT delete
# st.write("current state values from langraph are:", state)
# st.write("Next nodes:", snapshot.next)
# st.write("interrupts:", snapshot.interrupts)
# if interrupt_event:
#     st.write("Interrupt event is:", interrupt_event)

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

if (
    state.get("agents", {}).get("executor", {}).get("run_status") == "ok"
    and state.get("agents", {}).get("human_review", {}).get("final_decision") == "approve"
):
    # st.session_state.chat_history = state["messages"]
    st.success("Analysis completed")

    output = state.get("output", {})
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
