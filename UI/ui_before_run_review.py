import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)

def ui_before_run_review(app, config, payload, interrupt_id, queue_resume):
    ui_type = "before_run_review"
    st.subheader("🔍 Review Code Before Execution")
    if payload.get("code_summary"):
        st.caption("What this code does:")
        st.markdown(payload["code_summary"])
    if payload.get("code_assumptions"):
        st.caption("Assumptions:")
        st.markdown(payload["code_assumptions"])
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
            _dismiss_interrupt(interrupt_id)
            queue_resume(interrupt_id, {"action": "approve"})
            st.rerun()

        if go_regen:
            st.session_state[confirm_key] = False
            _dismiss_interrupt(interrupt_id)
            queue_resume(interrupt_id, {"action": "regenerate", "suggestion": suggestion})
            st.rerun()

        st.stop()
    
    col1, col2, col3 = st.columns(3)
    approve = col1.button("✅ Approve & Run", key = f"{ui_type}_approve")
    regenerate = col2.button("♻️ Regenerate Code", key = f"{ui_type}_regenerate")
    cancel = col3.button("❌ Cancel", key=f"{ui_type}_cancel")

    if cancel:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "cancel"})
        st.rerun()
    if approve:
        if suggestion:
            st.session_state[confirm_key] = True
            st.rerun()
        else:
            _dismiss_interrupt(interrupt_id)
            queue_resume(interrupt_id, {"action": "approve"})
            st.rerun()
    # ----------------------------
    # Regenerate: force suggestion
    # ----------------------------
    if regenerate:
        if not suggestion:
            st.error("Please enter an edit instruction before regenerating.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "regenerate", "suggestion": suggestion})
        st.rerun()
