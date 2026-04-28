"""Extract Agent: Small model extraction with Large model fallback."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.common import build_state_patch, extract_json_object, item_get, state_get
from agents.llm_router import call_chat_model
from prompts.extract_prompt import EXTRACT_PROMPT_TEMPLATE, NUMERICAL_EXTRACTION_PROMPT

try:
    from agents.mlx_extractor import MLXExtractor
    _MLX_AVAILABLE = True
except ImportError:
    MLXExtractor = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

_mlx_extractor: Optional[Any] = None


def _get_mlx_extractor() -> Optional[Any]:
    global _mlx_extractor
    if not _MLX_AVAILABLE:
        return None
    if _mlx_extractor is None:
        _mlx_extractor = MLXExtractor()
    return _mlx_extractor


def _normalize_picos(data: Dict[str, Any]) -> Dict[str, str]:
    return {
        "Population": str(data.get("Population", "")).strip(),
        "Intervention": str(data.get("Intervention", "")).strip(),
        "Comparison": str(data.get("Comparison", "")).strip(),
        "Outcome": str(data.get("Outcome", "")).strip(),
        "Study_Type": str(data.get("Study_Type", "")).strip(),
    }


def _extract_with_mlx(abstract: str) -> Optional[Dict[str, str]]:
    extractor = _get_mlx_extractor()
    if extractor is None:
        return None
    return extractor.extract(abstract)


def _extract_with_llm(abstract: str) -> Optional[Dict[str, str]]:
    prompt = EXTRACT_PROMPT_TEMPLATE.format(abstract=abstract)
    text = call_chat_model(prompt, temperature=0.0)
    data = extract_json_object(text)
    if not data:
        return None
    return _normalize_picos(data)


def _heuristic_extract(abstract: str) -> Dict[str, str]:
    s = abstract.strip().replace("\n", " ")
    return {
        "Population": s[:120],
        "Intervention": "",
        "Comparison": "",
        "Outcome": s[120:240],
        "Study_Type": "",
    }


def _validate_numerical(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean numerical extraction, flagging issues."""
    # Strip None values — pydantic str fields reject explicit None
    data = {k: v for k, v in data.items() if v is not None}

    issues = []
    es = data.get("effect_size")
    lo = data.get("ci_lower")
    hi = data.get("ci_upper")
    if es is not None and lo is not None and hi is not None:
        if not (lo < es < hi):
            issues.append("CI does not contain point estimate")
    for f in ["treatment_n", "control_n"]:
        v = data.get(f)
        if v is not None and (not isinstance(v, (int, float)) or v <= 0):
            issues.append(f"{f} invalid: {v}")

    conf = data.get("extraction_confidence", "MEDIUM")
    if issues:
        conf = "LOW"
    data["extraction_confidence"] = conf
    data["needs_review"] = (conf == "LOW")
    return data


def _extract_numerical(abstract: str) -> Optional[Dict[str, Any]]:
    """Extract numerical outcome data via DeepSeek (independent of PICOS)."""
    prompt = NUMERICAL_EXTRACTION_PROMPT.format(abstract=abstract)
    text = call_chat_model(prompt, temperature=0.0)
    data = extract_json_object(text)
    if not data:
        return None
    return _validate_numerical(data)


def extract_agent(state: Any) -> Dict[str, Any]:
    screened: List[Dict] = state_get(state, "screened_papers", []) or []
    extracted = []
    quantitative = []

    for paper in screened:
        abstract = item_get(paper, "abstract", "")
        pmid = str(item_get(paper, "pmid", ""))

        # ── PICOS文本提取（LoRA优先，DeepSeek回退） ──
        local = _extract_with_mlx(abstract)
        if local:
            source = "local"
            p = local
        else:
            llm = _extract_with_llm(abstract)
            if llm:
                source = "claude_or_glm"
                p = llm
            else:
                source = "heuristic"
                p = _heuristic_extract(abstract)

        extracted.append({
            "pmid": pmid,
            "population": p["Population"],
            "intervention": p["Intervention"],
            "comparison": p["Comparison"],
            "outcome": p["Outcome"],
            "study_type": p["Study_Type"],
            "extraction_source": source,
        })

        # ── 数值数据提取（独立于PICOS，DeepSeek） ──
        num = _extract_numerical(abstract)
        if num:
            num["pmid"] = pmid
            quantitative.append(num)

    return build_state_patch(
        extracted_picos=extracted,
        quantitative_outcomes=quantitative,
        error="",
    )
