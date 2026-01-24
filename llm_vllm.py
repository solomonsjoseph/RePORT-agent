from langchain_openai import ChatOpenAI
import os
import requests

def detect_vllm_model(base_url: str) -> str:
    url = base_url.rstrip("/") + "/models"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]

def build_llm(model_name, temperature=0.0, top_p=1.0, base_url=None, api_key=None):
    kwargs = dict(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
    )

    if base_url:
        # vLLM / OpenAI-compatible server
        kwargs["base_url"] = base_url
        kwargs["api_key"] = "dummy"
        # DO NOT set max_tokens
    else:
        # OpenAI
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        kwargs["max_tokens"] = 4096  # safe for OpenAI

    return ChatOpenAI(**kwargs)


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful data science assistant."),
#     ("user", "{question}")
# ])

# llm = ChatOpenAI(
#         base_url="http://localhost:8000/v1",
#         api_key="dummy",
#         model=model_name
#     )
# chain = prompt | llm

# result = chain.invoke({"question": "What is PCA in machine learning?"})
# print(result.content)


