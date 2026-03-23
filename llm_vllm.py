import os
import requests

# Info for vllm

DEFAULT_LLM_REQUEST_TIMEOUT_SEC = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "30"))

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
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
            timeout=DEFAULT_LLM_REQUEST_TIMEOUT_SEC,
            max_retries=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
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
