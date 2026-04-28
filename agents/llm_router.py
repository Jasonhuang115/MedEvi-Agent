"""LLM router - DeepSeek API with error resilience and fast-fail caching.

LangSmith tracing is automatically enabled when LANGCHAIN_TRACING_V2=true.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

_api_available: bool | None = None  # None = not checked, True/False = cached


def is_api_available() -> bool:
    """Check if DeepSeek API is configured and reachable. Cached after first check."""
    global _api_available
    if _api_available is not None:
        return _api_available

    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key or key.startswith("your_"):
        _api_available = False
        return False

    try:
        from tools.llm import DeepSeekLLM
        llm = DeepSeekLLM(temperature=0.0, max_tokens=16)
        llm.invoke("ping")
        _api_available = True
    except Exception:
        _api_available = False

    return _api_available


def _call_chat_model_impl(prompt: str, temperature: float = 0.0) -> str:
    """调用DeepSeek API，失败时返回空字符串。自动跳过不可用的API。"""
    if not is_api_available():
        return ""

    try:
        from tools.llm import DeepSeekLLM

        llm = DeepSeekLLM(temperature=temperature)
        return llm.invoke(prompt)
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return ""


# Conditionally wrap with LangSmith traceable decorator.
# This captures prompt, response, latency, and token usage automatically.
_call_chat_model_wrapped = _call_chat_model_impl

if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    try:
        from langsmith import traceable

        _call_chat_model_wrapped = traceable(
            _call_chat_model_impl,
            run_type="llm",
            name="DeepSeek",
            metadata={"provider": "deepseek"},
        )
        print("LangSmith tracing enabled for LLM calls")
    except Exception:
        pass


call_chat_model = _call_chat_model_wrapped
