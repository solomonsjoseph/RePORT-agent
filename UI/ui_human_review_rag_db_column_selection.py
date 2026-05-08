import streamlit as st

_RETRIEVAL_FALLBACK_NOTICE = (
    "Structured ranking output was unavailable. Showing retrieved candidate tables and columns directly for human review."
)


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


def _fallback_notice(rationale, fallback_reason):
    rationale_text = str(rationale or "").strip()
    fallback_text = str(fallback_reason or "").strip() or _RETRIEVAL_FALLBACK_NOTICE
    if fallback_text == rationale_text:
        return ""
    if rationale_text and fallback_text.startswith(f"{rationale_text} "):
        return fallback_text[len(rationale_text) :].strip()
    return fallback_text


def ui_human_review_rag_db_column_selection(app, config, payload, interrupt_id, queue_resume):
    ui_type = "human_review_rag_db_column_selection"

    st.subheader("🔍 Review DB-RAG Column Selection")
    st.caption("Approve the selected tables and columns before SQL generation.")

    goal_text = str(payload.get("goal_text") or "").strip()
    question = str(payload.get("question") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    selection_id = str(payload.get("selection_id") or "").strip()
    selection_source = str(payload.get("selection_source") or "").strip()
    fallback_reason = str(payload.get("fallback_reason") or "").strip()
    raw_model_output = str(payload.get("raw_model_output") or "").strip()
    columns = list(payload.get("columns") or [])
    feedback_history = list(payload.get("feedback_history") or [])

    if goal_text:
        st.markdown("**Interpreted extraction goal**")
        st.write(goal_text)
    if question:
        st.markdown("**Source question**")
        st.write(question)
    if selection_id:
        st.caption(f"Selection ID: {selection_id}")
    if rationale:
        st.markdown("**Rationale**")
        st.write(rationale)
    if selection_source == "retrieval_fallback":
        fallback_notice = _fallback_notice(rationale, fallback_reason)
        if fallback_notice:
            st.info(fallback_notice)
        if raw_model_output:
            with st.expander("Structured ranking raw output", expanded=False):
                st.code(raw_model_output, language="json")

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

    approve_col, revise_col, cancel_col = st.columns(3)
    approve = approve_col.button("✅ Approve", key=f"{ui_type}_approve_{interrupt_id}")
    regenerate = revise_col.button("♻️ Regenerate", key=f"{ui_type}_regenerate_{interrupt_id}")
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
