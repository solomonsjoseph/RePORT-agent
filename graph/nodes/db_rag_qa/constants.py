from __future__ import annotations

NODE_NAME = "rag_db_qa"
NODE_CAPABILITY = (
    "Handle database-grounded questions using retrieval over the local RePORT DB-RAG assets. "
    "Answer metadata questions from retrieved context, pause for human column review when SQL is "
    "needed, prepare read-only SQL only from approved selections, and keep prepared SQL candidates "
    "pending for explicit human review before execution."
)

SUPPORTED_PROVIDERS = {"openai", "anthropic"}
AFFIRMATIVE_REPLIES = {
    "y",
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please do",
    "do it",
    "go ahead",
}
NEGATIVE_REPLIES = {
    "n",
    "no",
    "nope",
    "nah",
    "not now",
    "don't",
    "do not",
}
NON_INFORMATIVE_FOLLOWUPS = {
    "k",
    "kk",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "thx",
    "got it",
    "sounds good",
    "please continue",
    "continue",
}
