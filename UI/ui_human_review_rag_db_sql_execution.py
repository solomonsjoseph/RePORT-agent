import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)


def ui_human_review_rag_db_sql_execution(app, config, payload, interrupt_id, queue_resume):
    ui_type = "human_review_rag_db_sql_execution"
    st.subheader("🧾 Review DB-RAG SQL Before Execution")
    st.caption("Approve the prepared SQL or regenerate it with feedback.")

    question = str(payload.get("question") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    selection_id = str(payload.get("selection_id") or "").strip()
    columns = list(payload.get("columns") or [])
    sql = str(payload.get("sql") or "").strip()
    feedback_history = list(payload.get("feedback_history") or [])

    if question:
        st.markdown("**Source question**")
        st.write(question)
    if selection_id:
        st.caption(f"Selection ID: {selection_id}")
    if rationale:
        st.markdown("**Selection rationale**")
        st.write(rationale)
    if columns:
        st.markdown("**Approved columns**")
        for column in columns:
            if isinstance(column, dict):
                table = str(column.get("table") or "").strip()
                column_name = str(column.get("column") or "").strip()
                description = str(column.get("description") or "").strip()
                label = f"`{table}.{column_name}`" if table and column_name else column_name or table
                if description:
                    label = f"{label}: {description}"
                st.write(f"- {label}")
            else:
                st.write(f"- {column}")
    if sql:
        st.markdown("**Prepared SQL**")
        st.code(sql, language="sql")
    if feedback_history:
        st.markdown("**Prior feedback history**")
        for idx, entry in enumerate(feedback_history, start=1):
            entry_dict = dict(entry or {}) if isinstance(entry, dict) else {}
            timestamp = str(entry_dict.get("timestamp") or entry_dict.get("created_at") or "").strip()
            action = str(entry_dict.get("action") or "").strip()
            feedback = str(entry_dict.get("feedback") or "").strip()
            label = f"{idx}. {timestamp}" if timestamp else f"{idx}."
            if action:
                label = f"{label} {action}"
            with st.expander(label, expanded=False):
                if feedback:
                    st.write(feedback)
                else:
                    st.write("No feedback text recorded.")

    feedback_key = f"feedback_{ui_type}_{interrupt_id}"
    feedback = st.text_area(
        "Optional suggestion / edit instruction",
        key=feedback_key,
        height=140,
    ).strip()

    confirm_key = f"confirm_approve_{ui_type}"

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
            queue_resume(interrupt_id, {"action": "regenerate", "suggestion": feedback})
            st.rerun()

        st.stop()

    approve_col, regenerate_col, cancel_col = st.columns(3)
    approve = approve_col.button("✅ Approve & Run", key=f"{ui_type}_approve_{interrupt_id}")
    regenerate = regenerate_col.button("♻️ Regenerate", key=f"{ui_type}_regenerate_{interrupt_id}")
    cancel = cancel_col.button("❌ Cancel", key=f"{ui_type}_cancel_{interrupt_id}")

    if cancel:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "cancel"})
        st.rerun()

    if approve:
        if feedback:
            st.session_state[confirm_key] = True
            st.rerun()
        else:
            _dismiss_interrupt(interrupt_id)
            queue_resume(interrupt_id, {"action": "approve"})
            st.rerun()

    if regenerate:
        if not feedback:
            st.error("Please enter a suggestion before regenerating.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "regenerate", "suggestion": feedback})
        st.rerun()
