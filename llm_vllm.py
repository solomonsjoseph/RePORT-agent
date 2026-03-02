from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
import requests
# Info for vllm
def detect_vllm_model(base_url: str) -> str:
    url = base_url.rstrip("/") + "/models"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]

def build_llm(model_name, temperature, top_p, base_url, api_key, provider):
    if provider == "vllm":
        # vLLM / OpenAI-compatible server
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            api_key="dummy",
        )

    if provider == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
        )

    if provider == "anthropic":
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
        )

    raise ValueError(f"Unsupported provider: {provider}")


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