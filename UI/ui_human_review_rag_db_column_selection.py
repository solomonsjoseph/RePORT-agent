import streamlit as st


def _dismiss_interrupt(interrupt_id):
    st.session_state["dismissed_interrupt_id"] = str(interrupt_id)


def _render_column_entry(column):
    if isinstance(column, dict):
        table = str(column.get("table") or "").strip()
        column_name = str(column.get("column") or "").strip()
        description = str(column.get("description") or "").strip()
        sample_values = column.get("sample_values")

        parts = []
        if table and column_name:
            parts.append(f"`{table}.{column_name}`")
        elif column_name:
            parts.append(f"`{column_name}`")
        elif table:
            parts.append(f"`{table}`")

        if description:
            parts.append(description)
        if sample_values:
            parts.append(f"Samples: {sample_values}")
        return " - ".join(parts) if parts else str(column)
    return str(column)


def ui_human_review_rag_db_column_selection(app, config, payload, interrupt_id, queue_resume):
    ui_type = "human_review_rag_db_column_selection"

    st.subheader("🔍 Review DB-RAG Column Selection")
    st.caption("Approve the selected tables and columns before SQL generation.")

    question = str(payload.get("question") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    selection_id = str(payload.get("selection_id") or "").strip()
    tables = list(payload.get("tables") or [])
    columns = list(payload.get("columns") or [])
    feedback_history = list(payload.get("feedback_history") or [])

    if question:
        st.markdown("**Source question**")
        st.write(question)
    if selection_id:
        st.caption(f"Selection ID: {selection_id}")
    if rationale:
        st.markdown("**Rationale**")
        st.write(rationale)

    if tables:
        st.markdown("**Selected tables**")
        for table in tables:
            st.write(f"- {table}")

    if columns:
        st.markdown("**Selected columns**")
        for column in columns:
            st.write(f"- {_render_column_entry(column)}")

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
        "Feedback for revised column selection",
        key=feedback_key,
        height=140,
    ).strip()

    approve_col, revise_col, cancel_col = st.columns(3)
    approve = approve_col.button("✅ Approve", key=f"{ui_type}_approve_{interrupt_id}")
    revise = revise_col.button("♻️ Revise", key=f"{ui_type}_revise_{interrupt_id}")
    cancel = cancel_col.button("✖️ Cancel", key=f"{ui_type}_cancel_{interrupt_id}")

    if approve:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "approve"})
        st.rerun()

    if revise:
        if not feedback:
            st.error("Please enter feedback before requesting a revision.")
            st.stop()
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "revise", "feedback": feedback})
        st.rerun()

    if cancel:
        _dismiss_interrupt(interrupt_id)
        queue_resume(interrupt_id, {"action": "cancel"})
        st.rerun()
