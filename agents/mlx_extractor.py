"""MLX Extractor: load fused LoRA model once, generate many times.

Replaces subprocess-based _extract_with_mlx() which loaded the 2.9GB model
from disk on every call. Now loads once and reuses for all papers.

Usage:
    extractor = MLXExtractor()
    picos = extractor.extract(abstract)  # first call loads model
    picos = extractor.extract(abstract2)  # reuses loaded model
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from prompts.extract_prompt import EXTRACT_PROMPT_TEMPLATE


class MLXExtractor:
    """Loads a fused LoRA model once and provides a generate interface."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or os.getenv(
            "MLX_PICOS_MODEL_PATH", "models/Qwen2.5-1.5B-Med-PICOS_v3"
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_available(self) -> bool:
        if self._loaded:
            return True
        return os.path.exists(self._model_path)

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True

        if not os.path.exists(self._model_path):
            return False

        with self._lock:
            if self._loaded:
                return True
            try:
                import mlx_lm

                self._model, self._tokenizer = mlx_lm.load(self._model_path)
                self._loaded = True
                return True
            except Exception:
                return False

    def extract(self, abstract: str, max_tokens: int = 256) -> Optional[Dict[str, str]]:
        if not self._ensure_loaded():
            return None

        prompt = EXTRACT_PROMPT_TEMPLATE.format(abstract=abstract)

        try:
            import mlx_lm

            output = mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        except Exception:
            return None

        data = _extract_json_from_output(output)
        if data is None:
            return None
        return _normalize_picos(data)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded = False


def _extract_json_from_output(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    import re

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


def _normalize_picos(data: Dict[str, Any]) -> Dict[str, str]:
    return {
        "Population": str(data.get("Population", "")).strip(),
        "Intervention": str(data.get("Intervention", "")).strip(),
        "Comparison": str(data.get("Comparison", "")).strip(),
        "Outcome": str(data.get("Outcome", "")).strip(),
        "Study_Type": str(data.get("Study_Type", "")).strip(),
    }
