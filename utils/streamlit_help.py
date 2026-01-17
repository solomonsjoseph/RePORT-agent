from langgraph.types import Command
from langchain_core.messages import HumanMessage
import streamlit as st

def ui_before_run_review(app, config, payload, interrupt_id):
    ui_type = "before_run_review"
    st.subheader("🔍 Review Code Before Execution")
    st.code(payload["generated_code"], language="python")

    confirm_key = f"confirm_approve_{ui_type}"
    suggestion_key = f"suggestion_{ui_type}_{interrupt_id}"

    suggestion = st.text_area(
        "Optional suggestion / edit instruction",
        key=suggestion_key,
        height=120,
    ).strip()

    # Confirmation UI (persistent via one flag)
    if st.session_state.get(confirm_key, False):
        st.warning("You typed a suggestion but clicked **Approve**. Proceed anyway?")
        c1, c2 = st.columns(2)
        proceed = c1.button("Proceed anyway", key=f"{ui_type}_proceed_{interrupt_id}")
        go_regen = c2.button("Regenerate instead", key=f"{ui_type}_regen_{interrupt_id}")

        if proceed:
            st.session_state[confirm_key] = False
            app.invoke(Command(resume={interrupt_id: {"action": "approve"}}), config=config)
            st.rerun()

        if go_regen:
            st.session_state[confirm_key] = False
            app.invoke(Command(resume={interrupt_id: {"action": "regenerate", "suggestion": suggestion}}), config=config)
            st.rerun()

        st.stop()
    
    col1, col2 = st.columns(2)
    approve = col1.button("✅ Approve & Run", key = f"{ui_type}_approve")
    regenerate = col2.button("♻️ Regenerate Code", key = f"{ui_type}_regenerate")

    if approve:
        if suggestion:
            st.session_state[confirm_key] = True
            st.rerun()
        else:
            app.invoke(
                Command(resume={interrupt_id: {"action": "approve"}}),
                config=config,
            )
            st.rerun()
    # ----------------------------
    # Regenerate: force suggestion
    # ----------------------------
    if regenerate:
        if not suggestion:
            st.error("Please enter an edit instruction before regenerating.")
            st.stop()
        app.invoke(
            Command(resume={
                interrupt_id: {
                    "action": "regenerate",
                    "suggestion": suggestion
                }
            }),
            config=config
        )
        st.rerun()

def ui_final_review(app, config, payload, interrupt_id):
    ui_type = "final_review"
    st.subheader("✅ Final Review")

    st.success("Execution succeeded")
    if payload["generated_code"]:
        st.markdown("Generated Code:")
        st.code(payload["generated_code"], language="python")
    if payload["output"]:
        st.text("Output:")
        st.code(payload["output"], language="text")
    if payload['figure_png']:
        st.text("Generated Figure:")
        st.image(payload['figure_png'])

    suggestion = st.text_area(
        "Optional suggestion / edit instruction",
        key=f"suggestion_{ui_type}",
        height=120,
    )
    col1, col2 = st.columns(2)
    approve = col1.button("✅ Approve & Finish", key = f"{ui_type}_approve")
    regenerate = col2.button("♻️ Regenerate", key = f"{ui_type}_regenerate")

    if approve:
        # st.session_state[f"handled_{interrupt_id}"] = True
        app.invoke(
            Command(resume={
                interrupt_id: {
                    "action": "approve"
                }
            }),
            config=config
        )
        st.rerun()
    if regenerate:
        # st.session_state.chat_history.append(
        #     HumanMessage(content = suggestion if suggestion else "User requests code edit / regeneration.")
        # )
        app.invoke(
            Command(resume={
                interrupt_id: {
                    "action": "regenerate",
                    "suggestion": suggestion
                }
            }),
            config=config
        )
        # st.session_state["human_suggestion_box"] = ""
        st.rerun()