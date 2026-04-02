import os
import requests
from utils.openai_models import is_openai_gpt5_family

# Info for vllm

DEFAULT_LLM_REQUEST_TIMEOUT_SEC = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "60"))

def _openai_chat_kwargs(model_name, temperature, top_p):
    kwargs = {
        "model": model_name,
        "timeout": DEFAULT_LLM_REQUEST_TIMEOUT_SEC,
        "max_retries": 0,
    }

    # GPT-5 family models do not share the same sampling parameter support as
    # GPT-4o-style chat models. Keep the request conservative to avoid 400s
    # from unsupported parameters such as top_p.
    if is_openai_gpt5_family(model_name):
        kwargs["max_completion_tokens"] = 4096
        return kwargs

    kwargs.update(
        {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 4096,
        }
    )
    return kwargs

def detect_vllm_model(base_url: str) -> str:
    url = base_url.rstrip("/") + "/models"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def build_llm(model_name, temperature, top_p, base_url, api_key, provider):
    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            api_key="dummy",
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(**_openai_chat_kwargs(model_name, temperature, top_p))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=4096,
            timeout=DEFAULT_LLM_REQUEST_TIMEOUT_SEC,
            max_retries=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=4096,
            timeout=DEFAULT_LLM_REQUEST_TIMEOUT_SEC,
            max_retries=0,
        )

    raise ValueError(f"Unsupported provider: {provider}")
