"""Shared utilities for agent nodes."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


def state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def item_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from a dict OR pydantic object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_state_patch(**kwargs: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None

    return None


def safe_float(value: Any, default: float = 0.5) -> float:
    try:
        out = float(value)
        return max(0.0, min(1.0, out))
    except Exception:
        return default


def has_deepseek_ready() -> bool:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    return bool(key) and not key.startswith("your_")


def to_dict(obj: Any) -> Any:
    """Recursively convert pydantic models to plain dicts for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def normalize_pico_query(pico_query: Any) -> Dict[str, str]:
    """Normalize pico_query to a plain dict (handles pydantic PICOQuery)."""
    if pico_query is None:
        return {}
    if isinstance(pico_query, dict):
        return pico_query
    return pico_query.model_dump() if hasattr(pico_query, "model_dump") else {}
