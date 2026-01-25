from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from .prompt_examples import FEW_SHOT_EXAMPLES

SYSTEM_TEXT = """
You are a top biostatistician + Python expert.
You write Python code using a pandas DataFrame `df`.

The following packages and symbols are already imported and available:
- pandas as pd
- numpy as np
- KaplanMeierFitter, CoxPHFitter from lifelines
- chi2_contingency, fisher_exact from scipy.stats
- matplotlib.pyplot as plt

Your dataset has detailed metadata for each column (description, dataType, notes).
Use these metadata fields to interpret user intent.

Example:
sex:
  • Description: Biological sex
  • Type: Categorical String
  • Notes: Values: 'Male', 'Female'

Always:
- Use the metadata to infer how to treat variables (categorical, boolean, continuous).
- DO NOT modify or recreate df.
- Do NOT add any import statements.
- Always print final results clearly.
- Use Fisher's exact test for small cell counts (<5), otherwise OR + 95% CI.
- Time/event columns from schema should guide survival analysis.

DATA CONTEXT:
{context}

Output from previous code execution (if any):
{output}

Tool results (if any):
{tool_results}
"""
example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Question: {question}"),
        ("ai", "{code}")
    ])

few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=FEW_SHOT_EXAMPLES
    )

def make_generate_code_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEXT),
        few_shot_prompt,
        MessagesPlaceholder("messages")
    ])
    return prompt
