import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)


def _error_review_message(error_payload: dict | None) -> str:
    category = (error_payload or {}).get("category")
    if category == "policy_blocked":
        return "Sandbox policy blocked this code. The app will not auto-retry."
    if category == "unsupported_runtime":
        return "The requested package/runtime capability is not available in the sandbox image."
    if category == "timeout":
        return "The analysis exceeded the sandbox time limit and was not retried automatically."
    if category == "infrastructure":
        return "The sandbox runner failed due to an environment/runtime issue."
    return "The assistant hit repeated errors while executing the code."


def ui_after_error_review(app, config, payload, interrupt_id, queue_resume):
    ui_type = "after_error_review"
    st.subheader("⚠️ Error Resolution Needed")
    error_payload = payload.get("error") or {}
    st.warning(_error_review_message(error_payload))
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

    submit_col, cancel_col = st.columns(2)
    submit = submit_col.button("Submit feedback", key=f"{ui_type}_submit")
    cancel = cancel_col.button("❌ Cancel", key=f"{ui_type}_cancel")

    if cancel:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "cancel"})
        st.rerun()

    if submit:
        if not suggestion:
            st.error("Please enter feedback before continuing.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "feedback", "suggestion": suggestion})
        st.rerun()
