import streamlit as st
import pandas as pd
import json
import os
import hashlib
import io
import tempfile
import requests
from pathlib import Path
# from pathlib import Path
import uuid
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from graph.builder import build_graph
from llm_vllm import build_llm, detect_vllm_model
from UI.ui_before_run_review import ui_before_run_review
from UI.ui_after_error_review import ui_after_error_review
from UI.ui_final_review import ui_final_review
from UI.ui_human_review_rag_db_column_selection import ui_human_review_rag_db_column_selection
from UI.ui_human_review_rag_db_sql_execution import ui_human_review_rag_db_sql_execution
from UI.load_openai import load_openai
from UI.load_anthropic import load_anthropic
from db_rag.config import (
    resolve_db_rag_embedding_model,
    resolve_db_rag_reranker_model,
)
from utils.openai_models import list_supported_openai_chat_models
from utils.execution_mode import (
    apply_execution_mode,
    allow_trusted_local_policy_blocked,
    current_execution_mode,
    docker_available,
)
from utils.streamlit_config import (
    ANTHROPIC_API_VERSION,
    DEFAULT_ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_EXECUTION_TIMEOUT_SEC,
    DEFAULT_MAX_AUTO_STEPS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    EXECUTION_MODE_OPTIONS,
    EXECUTION_TIMEOUT_RANGE,
    EXECUTION_TIMEOUT_STEP,
    MAX_AUTO_STEPS_RANGE,
    PROVIDER_OPTIONS,
    provider_display_label,
    TEMPERATURE_RANGE,
    TEMPERATURE_STEP,
    TOP_P_RANGE,
    TOP_P_STEP,
)
from utils.run_manager import GraphRunManager
from utils.export_thread import build_thread_export
from utils.message_window import compact_messages
from utils.display_history import build_display_history, serialize_display_history
from graph.conversation_events import ensure_conversation_state
from graph.state_views import get_conversation_events
from utils.streamlit_interrupts import (
    blocking_review_notice,
    should_block_chat_submission,
    should_render_review_interrupt,
)
from utils.streamlit_rendering import (
    canonicalize_conversation_history,
    conversation_controls_state,
    conversation_history_with_pending_user,
    normalize_submitted_question,
)
from utils.dataset_artifacts import (
    build_active_dataset_artifacts_patch,
    build_dataset_artifacts_patch,
    load_dataset_artifact,
    persist_dataset_artifact,
)
from utils.message_attachments import build_user_message_attachments

WELCOME_MESSAGE = "Hello! Ask me anything ..."
# --------------------------
# Streamlit Config
# --------------------------
title = "AI Agent for RePORT"
st.set_page_config(
    page_title=title,
    layout="wide",
)

load_dotenv()

st.title(title)

# ============================================================
# Sidebar UI — Model Configuration
# ============================================================

st.sidebar.header("⚙️ Model Settings")
base_url = DEFAULT_BASE_URL

# default_model = 'meta-llama/Llama-3.1-8B-Instruct'
default_temp = DEFAULT_TEMPERATURE
default_api_key = DEFAULT_API_KEY
default_openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
default_anthropic_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
openai_env_key = os.getenv("OPENAI_API_KEY", "")
anthropic_env_key = os.getenv("ANTHROPIC_API_KEY", "")

try:
    active_db_rag_embedding_model = resolve_db_rag_embedding_model()
    db_rag_embedding_error = ""
except Exception as exc:
    active_db_rag_embedding_model = None
    db_rag_embedding_error = str(exc)

try:
    active_db_rag_reranker_model = resolve_db_rag_reranker_model()
    db_rag_reranker_error = ""
except Exception as exc:
    active_db_rag_reranker_model = None
    db_rag_reranker_error = str(exc)

provider = st.sidebar.selectbox(
    "Provider",
    list(PROVIDER_OPTIONS),
    index=PROVIDER_OPTIONS.index(DEFAULT_PROVIDER),
    format_func=provider_display_label,
    help="Choose the model provider to use."
)

show_debug_state = st.sidebar.toggle(
    "🐛 Show debug state",
    value=False,
)
max_auto_steps = st.sidebar.slider(
    "Auto-run steps per refresh",
    min_value=MAX_AUTO_STEPS_RANGE[0],
    max_value=MAX_AUTO_STEPS_RANGE[1],
    value=DEFAULT_MAX_AUTO_STEPS,
    help="Number of workflow nodes the app runs in the background before the next UI refresh. Higher values feel faster but intermediate steps may be less visible.",
)
execution_timeout = st.sidebar.slider(
    "Execution timeout (seconds)",
    min_value=EXECUTION_TIMEOUT_RANGE[0],
    max_value=EXECUTION_TIMEOUT_RANGE[1],
    value=DEFAULT_EXECUTION_TIMEOUT_SEC,
    step=EXECUTION_TIMEOUT_STEP,
)

execution_mode = st.sidebar.selectbox(
    "Execution mode",
    list(EXECUTION_MODE_OPTIONS),
    index=EXECUTION_MODE_OPTIONS.index(current_execution_mode()),
    help="Choose how approved Python code is executed.",
)

apply_execution_mode(execution_mode)

allow_policy_blocked_trusted_local = st.sidebar.checkbox(
    "Allow policy-blocked operations in trusted local",
    value=allow_trusted_local_policy_blocked() if "ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED" in os.environ else DEFAULT_ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED,
    help=(
        "Only affects trusted local execution. When enabled, trusted-local runs "
        "may proceed past filesystem-mutation policy checks instead of stopping "
        "immediately."
    ),
    disabled=execution_mode != "trusted_local",
)
os.environ["ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED"] = (
    "1" if allow_policy_blocked_trusted_local else "0"
)

if docker_available():
    st.sidebar.caption("Docker detected on PATH.")
else:
    st.sidebar.warning("Docker is not available on PATH. Stick to `trsusted_local`.")
    if execution_mode == "docker":
        st.sidebar.info("Switch to `trusted_local` to run code without Docker.")

os.environ["EXECUTION_TIMEOUT_SEC"] = str(execution_timeout)

api_key = ""
model_name = ""

@st.cache_data(show_spinner=False)
def load_openai_models(effective_api_key):
    return list_supported_openai_chat_models(effective_api_key)

@st.cache_data(show_spinner=False)
def load_anthropic_models(effective_api_key):
    headers = {
        "x-api-key": effective_api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
    }
    resp = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    model_ids = sorted({item.get("id") for item in data if item.get("id")})
    return model_ids

if provider == "vllm":
    try:
        model_name = detect_vllm_model(base_url)
    except Exception:
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
elif provider == "anthropic":
    api_key, model_name = load_anthropic(
        default_api_key=default_api_key,
        anthropic_env_key=anthropic_env_key,
        default_anthropic_model=default_anthropic_model,
        load_anthropic_models_fn=load_anthropic_models,
    )

temperature = st.sidebar.slider(
    "Temperature",
    min_value=TEMPERATURE_RANGE[0],
    max_value=TEMPERATURE_RANGE[1],
    value=default_temp,
    step=TEMPERATURE_STEP,
    help="Higher temperature = more creative code."
)

top_p = st.sidebar.slider(
    "Top probability",
    min_value=TOP_P_RANGE[0],
    max_value=TOP_P_RANGE[1],
    value=DEFAULT_TOP_P,
    step=TOP_P_STEP,
    help="Set the top-p value, lowering it increases creativity."
)
if provider == "anthropic":
    st.sidebar.caption("Anthropic models in this app use temperature only; top-p is ignored.")

st.sidebar.divider()
st.sidebar.markdown("**DB-RAG Runtime**")
if db_rag_embedding_error:
    st.sidebar.error(f"Embedding index: {db_rag_embedding_error}")
else:
    st.sidebar.caption("Embedding index")
    st.sidebar.code(str(active_db_rag_embedding_model), language=None)

if db_rag_reranker_error:
    st.sidebar.error(f"Column reranker: {db_rag_reranker_error}")
elif active_db_rag_reranker_model:
    st.sidebar.caption("Column reranker")
    st.sidebar.code(str(active_db_rag_reranker_model), language=None)
else:
    st.sidebar.caption("Column reranker")
    st.sidebar.code("disabled", language=None)
    st.sidebar.caption("Set `DB_RAG_RERANKER_MODEL` in `.env` to enable reranking in the app.")

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
def load_app(llm, provider, dataset_signature):
    db_path = os.path.join(tempfile.gettempdir(), "report-agent", "agent_memory.db")
    return build_graph(llm, provider, db_path=db_path)


llm = load_llm(model_name, temperature, top_p, base_url, api_key, provider)
app = load_app(llm, provider, dataset_signature)

@st.cache_resource
def load_run_manager():
    return GraphRunManager()

run_manager = load_run_manager()

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
    st.session_state.chat_history = [AIMessage(content=WELCOME_MESSAGE)]

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = dataset_signature
elif st.session_state.dataset_signature != dataset_signature:
    st.session_state.dataset_signature = dataset_signature
    st.info("Detected new dataset/schema. Previous conversation is preserved. Use 'Reset Conversation' to clear history.")
    st.rerun()

backend_signature = f"{provider}:{model_name}"
if "backend_signature" not in st.session_state:
    st.session_state.backend_signature = backend_signature
elif st.session_state.backend_signature != backend_signature:
    st.session_state.backend_signature = backend_signature
    st.session_state.chat_history = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state.thread_id = uuid.uuid4().hex
    st.info("Model backend changed. Started a fresh conversation thread to avoid stale orchestrator state.")
    st.rerun()

# ============================================================
# 4. Chat Input
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

def initial_graph_state(user_message: HumanMessage, uploaded_artifact: dict | None) -> dict:
    """Bootstrap state for the first turn of a new thread.

    Later turns should send only message deltas so checkpointed graph state is preserved.
    """
    return ensure_conversation_state({
        "messages": [user_message],
        "output": {},
        "artifacts": {
            "datasets": ({uploaded_artifact["id"]: uploaded_artifact} if uploaded_artifact else {}),
            "active_dataset_id": uploaded_artifact["id"] if uploaded_artifact else None,
        },
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
            "rag_db_qa": {},
            "generate_code": {},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": [],
            "thread_id": st.session_state.thread_id,
        },
    })


def next_turn_payload(
    user_message: HumanMessage,
    uploaded_artifact: dict | None = None,
    checkpoint_state: dict | None = None,
) -> dict:
    """Send only the new user message so checkpointed graph state is preserved.

    The orchestrator itself is responsible for resetting transient turn state on
    genuine new turns. Passing a full replacement payload here destroys
    clarification context such as pending_question and clarification_return_node.
    """
    payload = {
        "messages": [user_message],
    }
    if uploaded_artifact:
        payload["artifacts"] = build_dataset_artifacts_patch(
            (checkpoint_state or {}).get("artifacts"),
            uploaded_artifact,
        )
    return payload

snapshot = app.get_state(config)
has_graph_state = bool(snapshot and snapshot.values)

def queue_interrupt_resume(interrupt_id, payload):
    st.session_state["pending_interrupt_resume"] = {
        "interrupt_id": str(interrupt_id),
        "payload": payload,
    }


def queue_question_submission():
    text = normalize_submitted_question(
        st.session_state.get("question_input", ""),
        blocked=bool(st.session_state.get("chat_submission_blocked", False)),
    )
    if text:
        st.session_state["pending_question_text"] = text


def set_active_dataset_selection(app, config, state: dict, dataset_id: str) -> None:
    artifacts = build_active_dataset_artifacts_patch(state.get("artifacts"), dataset_id)
    app.update_state(config, {"artifacts": artifacts})

# --------------------------------------------------
# Run LangGraph (one step)
# --------------------------------------------------
snapshot = app.get_state(config)
state = snapshot.values if snapshot else {}

pending_resume = st.session_state.pop("pending_interrupt_resume", None)
if pending_resume:
    app.invoke(
        Command(resume={pending_resume["interrupt_id"]: pending_resume["payload"]}),
        config=config,
    )
    snapshot = app.get_state(config)
    state = snapshot.values if snapshot else {}

pending_question_text = st.session_state.pop("pending_question_text", None)
pending_user_message = None
if pending_question_text:
    uploaded_artifact = None
    if uploaded_csv and uploaded_schema:
        uploaded_artifact = persist_dataset_artifact(
            runtime_root=None,
            thread_id=st.session_state.thread_id,
            dataset_id=f"uploaded-{dataset_signature[:8]}",
            kind="uploaded",
            dataframe=df,
            schema=schema,
            provenance={"source": "upload", "dataset_signature": dataset_signature},
        )
    user_message = HumanMessage(
        content=pending_question_text,
        additional_kwargs={"attachments": build_user_message_attachments(uploaded_artifact)},
    )

    if has_graph_state:
        initial_payload = next_turn_payload(user_message, uploaded_artifact, snapshot.values)
    else:
        initial_payload = initial_graph_state(user_message, uploaded_artifact)

    submitted = run_manager.submit(
        thread_id=st.session_state.thread_id,
        app=app,
        config=config,
        max_steps=max_auto_steps,
        initial_payload=initial_payload,
    )
    if not submitted:
        st.session_state["pending_submission_warning"] = "A run is already in progress. Please wait..."
    else:
        pending_user_message = user_message

conversation_placeholder = st.empty()
# for msg in st.session_state.chat_history:
#     with st.chat_message(
#         "user" if isinstance(msg, HumanMessage) else "assistant"
#     ):
#         st.markdown(msg.content)

# For check workflow state, DEBUG ONLY
# with st.expander("🧭 Current workflow state", expanded=False):
#     run_status = run_manager.status(st.session_state.thread_id)
#     executor_state = state.get("agents", {}).get("executor", {})
#     review_state = state.get("agents", {}).get("human_review", {})
#     st.write(
#         {
#             "background_run_state": run_status.get("state"),
#             "background_run_steps": run_status.get("steps"),
#             "background_run_error": run_status.get("error"),
#             "last_action": state.get("last_action"),
#             "next_action": state.get("next_action"),
#             "next_nodes": list(snapshot.next or []) if snapshot else [],
#             "executor_run_status": executor_state.get("run_status"),
#             "before_run_decision": review_state.get("before_run_decision"),
#             "after_error_decision": review_state.get("after_error_decision"),
#             "final_decision": review_state.get("final_decision"),
#             "workflow_trace_tail": list(state.get("meta", {}).get("workflow_trace", []))[-12:],
#         }
#     )

if state:
    st.session_state.chat_history = build_display_history(state)
st.session_state.chat_history = conversation_history_with_pending_user(
    st.session_state.chat_history,
    pending_user_message,
    welcome_message=WELCOME_MESSAGE,
)
st.session_state.chat_history = canonicalize_conversation_history(
    st.session_state.chat_history,
    welcome_message=WELCOME_MESSAGE,
)

latest_chat_is_human = bool(st.session_state.chat_history) and isinstance(
    st.session_state.chat_history[-1], HumanMessage
)
output = {} if latest_chat_is_human else (state.get("output", {}) if state else {})
export_bytes = build_thread_export(
    thread_id=st.session_state.thread_id,
    provider=provider,
    model_name=model_name,
    state={**state, "output": output} if state else {"output": output, "messages": st.session_state.chat_history},
)
executor_ok = state.get("agents", {}).get("executor", {}).get("run_status") == "ok" if state else False
meta = state.get("meta", {}) if state else {}
current_code_hash = meta.get("current_code_hash")
final_approved_code_hash = meta.get("final_approved_code_hash")
final_approved = bool(
    current_code_hash
    and final_approved_code_hash
    and current_code_hash == final_approved_code_hash
) if state else False
qa_ready = bool(output.get("qa_response"))
analysis_ready = executor_ok and final_approved

# Render all previous chat history
display_history = list(st.session_state.chat_history)

interrupt_event = snapshot.interrupts[0] if snapshot and snapshot.interrupts else None
dismissed_interrupt_id = str(st.session_state.get("dismissed_interrupt_id", "") or "")
review_state = state.get("agents", {}).get("human_review", {}) if state else {}

# Clear stale dismissal marker once there is no active interrupt.
if not interrupt_event and dismissed_interrupt_id:
    st.session_state.pop("dismissed_interrupt_id", None)
    dismissed_interrupt_id = ""

if show_debug_state:
    st.write("current state values from langraph are:", state)
    if state:
        with st.expander("Rendered conversation history", expanded=False):
            st.write(serialize_display_history(build_display_history(state)))
        with st.expander("Semantic conversation events", expanded=False):
            st.write(get_conversation_events(state))
    st.write("Next nodes:", snapshot.next)
    st.write("interrupts:", snapshot.interrupts)
    if interrupt_event:
        st.write("Interrupt event is:", interrupt_event)

should_render_interrupt = should_render_review_interrupt(
    interrupt_event,
    dismissed_interrupt_id=dismissed_interrupt_id,
    review_state=review_state,
)
interrupt_id = None
payload = None
ui_type = None
if should_render_interrupt:
    interrupt_id = interrupt_event.id
    payload = interrupt_event.value
    ui_type = payload["type"]

    # --------------------------------------------------------
    # Review BEFORE execution
    # --------------------------------------------------------
if should_render_interrupt:
    if ui_type == "before_run_review":
        ui_before_run_review(app, config, payload, interrupt_id, queue_interrupt_resume)
    elif ui_type == "after_error_review":
        ui_after_error_review(app, config, payload, interrupt_id, queue_interrupt_resume)
    elif ui_type == "human_review_rag_db_column_selection":
        ui_human_review_rag_db_column_selection(app, config, payload, interrupt_id, queue_interrupt_resume)
    elif ui_type == "human_review_rag_db_sql_execution":
        ui_human_review_rag_db_sql_execution(app, config, payload, interrupt_id, queue_interrupt_resume)
    elif ui_type == "final_review":
        ui_final_review(app, config, payload, interrupt_id, queue_interrupt_resume)

run_status = run_manager.status(st.session_state.thread_id)
if snapshot and snapshot.next and not snapshot.interrupts and not run_manager.is_running(st.session_state.thread_id):
    run_manager.submit(
        thread_id=st.session_state.thread_id,
        app=app,
        config=config,
        max_steps=max_auto_steps,
        initial_payload=None,
    )
    run_status = run_manager.status(st.session_state.thread_id)

# Render the conversation in a single replaceable block to prevent stale duplicate
# headers/messages across rapid reruns while background work is active.
with conversation_placeholder.container():
    st.subheader("💬 Conversation", anchor=False)
    for msg in display_history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

            if isinstance(msg, AIMessage):
                fig_path = (msg.additional_kwargs or {}).get("figure_path")
                if fig_path:
                    st.image(fig_path)
                    st.download_button(
                        label="⬇️ Download plot (PNG)",
                        data=Path(fig_path).read_bytes(),
                        file_name="plot.png",
                        mime="image/png",
                        key=f"dl_{id(msg)}",
                    )

    if run_status.get("state") == "running":
        st.info("⏳ Working in background...")
    elif run_status.get("state") == "error":
        st.error(f"Background workflow failed: {run_status.get('error') or 'unknown error'}")

review_blocks_submission = should_block_chat_submission(
    interrupt_event,
    dismissed_interrupt_id=dismissed_interrupt_id,
    review_state=review_state,
)
controls_state = conversation_controls_state(
    run_status,
    review_blocked=review_blocks_submission,
)
run_in_progress = controls_state.run_in_progress

if qa_ready and not analysis_ready:
    st.success("Response ready")

if analysis_ready:
    st.success("Analysis completed")

if state:
    artifacts = state.get("artifacts", {}) or {}
    datasets = artifacts.get("datasets", {}) or {}
    active_dataset_id = artifacts.get("active_dataset_id")
    dataset_ids = list(datasets.keys())

    if dataset_ids:
        default_index = dataset_ids.index(active_dataset_id) if active_dataset_id in dataset_ids else 0
        with st.expander("💾 Saved Datasets", expanded=False):
            selected_dataset_id = st.selectbox(
                "Active dataset artifact",
                options=dataset_ids,
                index=default_index,
                key="selected_dataset_artifact_id",
            )
            if selected_dataset_id != active_dataset_id:
                set_active_dataset_selection(app, config, state, selected_dataset_id)
                st.rerun()
            selected_artifact = datasets.get(selected_dataset_id)
            if selected_artifact:
                st.caption(f"Dataset ID: {selected_artifact.get('id', 'unknown')}")
                st.caption(f"Kind: {selected_artifact.get('kind', 'unknown')}")
                if selected_dataset_id == active_dataset_id:
                    st.caption("Status: active")
                try:
                    selected_df, selected_schema = load_dataset_artifact(selected_artifact)
                    st.dataframe(selected_df.head(100))
                    csv_bytes = selected_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download selected dataset (CSV)",
                        data=csv_bytes,
                        file_name=f"{selected_artifact.get('id', 'dataset')}.csv",
                        mime="text/csv",
                        key=f"download_dataset_csv_{selected_artifact.get('id', 'unknown')}",
                    )
                    with st.expander("Schema (selected dataset)", expanded=False):
                        st.json(selected_schema)
                except Exception as exc:
                    st.error(f"Unable to load selected dataset artifact: {exc}")

with st.container():
    chat_submission_blocked = controls_state.submission_blocked
    st.session_state["chat_submission_blocked"] = chat_submission_blocked
    pending_submission_warning = st.session_state.pop("pending_submission_warning", None)
    if pending_submission_warning:
        st.warning(pending_submission_warning)
    review_notice = blocking_review_notice(
        interrupt_event,
        dismissed_interrupt_id=dismissed_interrupt_id,
        review_state=review_state,
    )
    if review_notice:
        st.info(review_notice)
    with st.form("question_form", clear_on_submit=True):
        st.text_input(
            "Ask a question about your dataset!",
            placeholder="Ask a question about your dataset!",
            label_visibility="collapsed",
            disabled=chat_submission_blocked,
            key="question_input",
        )
        st.form_submit_button(
            "Send",
            disabled=chat_submission_blocked,
            on_click=queue_question_submission,
        )

    action_col, save_col = st.columns([1, 1])
    with action_col:
        if st.button("🔄 Reset Conversation"):
            st.session_state.chat_history = [AIMessage(content=WELCOME_MESSAGE)]
            st.session_state.thread_id = uuid.uuid4().hex
            st.rerun()
    with save_col:
        st.download_button(
            label="💾 Save Current Thread",
            data=export_bytes,
            file_name=f"thread_{st.session_state.thread_id}.zip",
            mime="application/zip",
            key="save_current_thread_bottom",
            help="Download this thread's conversation, generated code, output text, and figure as a ZIP archive.",
        )

if run_status.get("state") == "running":
    time.sleep(0.25)
    st.rerun()
    st.stop()
elif run_status.get("state") == "error":
    st.stop()

if should_render_interrupt:
    st.stop()
