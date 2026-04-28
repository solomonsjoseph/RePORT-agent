NODE_NAME = "rag_db_qa"
NODE_CAPABILITY = (
    "Handle database-grounded questions using retrieval over the local RePORT DB-RAG assets. "
    "Answer metadata questions from retrieved context, pause for human column review when SQL is "
    "needed, prepare read-only SQL only from approved selections, and keep prepared SQL candidates "
    "pending for explicit human review before execution."
)

from .db_rag_qa import (
    _deserialize_prepared_sql_candidate,
    _execute_prepared_sql_candidate,
    rag_db_qa_node,
)

__all__ = [
    "NODE_NAME",
    "NODE_CAPABILITY",
    "rag_db_qa_node",
    "_deserialize_prepared_sql_candidate",
    "_execute_prepared_sql_candidate",
]
