from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_FIX = """
You are a debugging assistant.
Using the column metadata (description, dataType, notes), fix the previous code.

Important:
- Treat Boolean Strings ('Yes'/'No') as categorical variables.
- Treat categorical columns exactly as described in the schema.
- Use time/event variables ONLY when schema indicates they are relevant.
- Do NOT add imports. Use the selected dataset as datasets["<dataset_id>"], using the dataset ID shown in DATA CONTEXT.
- Always output ONE corrected python code block.

DATA CONTEXT:
{context}
"""

def make_fix_code_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_FIX),
            ("system", "Previous code:\n```python\n{code}\n```"),
            ("system", "Error message:\n{error_message}"),
            ("system", "Error type:\n{error_type}"),
            MessagesPlaceholder("messages"),
        ]
    )
