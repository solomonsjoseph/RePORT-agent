from langgraph.types import Command
import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)


def ui_after_error_review(app, config, payload, interrupt_id):
    ui_type = "after_error_review"
    st.subheader("⚠️ Error Resolution Needed")
    st.warning("The assistant hit repeated errors while executing the code.")

    error_payload = payload.get("error") or {}
    if error_payload:
        st.markdown("Error details:")
        st.code(
            f"{error_payload.get('type', 'UnknownError')}: {error_payload.get('message', '')}",
            language="text",
        )

    if payload.get("generated_code"):
        st.markdown("Last generated code:")
        st.code(payload["generated_code"], language="python")

    suggestion_key = f"suggestion_{ui_type}_{interrupt_id}"
    suggestion = st.text_area(
        "Provide feedback or new instructions for the next attempt",
        key=suggestion_key,
        height=140,
    ).strip()

    if st.button("Submit feedback", key=f"{ui_type}_submit"):
        if not suggestion:
            st.error("Please enter feedback before continuing.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        app.invoke(
            Command(resume={interrupt_id: {"action": "feedback", "suggestion": suggestion}}),
            config=config,
        )
        st.rerun()
