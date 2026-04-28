"""Screen Agent: PICOS-aware include/exclude with reflection LLM."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agents.common import build_state_patch, normalize_pico_query, safe_float, state_get
from agents.llm_router import call_chat_model
from prompts.screen_prompt import REFLECTION_PROMPT_TEMPLATE, SCREEN_PROMPT_TEMPLATE


def _parse_screen_output(text: str) -> Tuple[str, str, float]:
    decision = "Exclude"
    reason = "未满足纳入标准"
    confidence = 0.5

    if not text:
        return decision, reason, confidence

    d = re.search(r"Decision:\s*(Include|Exclude)", text, flags=re.I)
    r = re.search(r"Reason:\s*(.+)", text)
    c = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text)

    if d:
        decision = d.group(1).capitalize()
    if r:
        reason = r.group(1).strip()
    if c:
        confidence = safe_float(c.group(1), 0.5)

    return decision, reason, confidence


def _fallback_screen(query: str, abstract: str) -> Tuple[str, str, float]:
    q_terms = [t.lower() for t in query.split() if len(t) >= 3]
    a = abstract.lower()
    overlap = sum(1 for t in q_terms if t in a)

    if overlap >= max(2, len(q_terms) // 3):
        return "Include", "与检索词语义相关（fallback规则）", 0.55
    return "Exclude", "与检索词相关性不足（fallback规则）", 0.55


def screen_agent(state: Any) -> Dict[str, Any]:
    query = state_get(state, "query", "")
    pico_query = normalize_pico_query(state_get(state, "pico_query", {}))
    docs: List[Dict] = state_get(state, "reranked_abstracts", []) or []

    # ── 第一轮筛选 ──
    screened = []
    excluded = []
    all_results = []

    for doc in docs:
        abstract = doc.get("abstract", "")
        prompt = SCREEN_PROMPT_TEMPLATE.format(
            population=pico_query.get("population", ""),
            intervention=pico_query.get("intervention", ""),
            comparison=pico_query.get("comparison", ""),
            outcome=pico_query.get("outcome", ""),
            study_type=pico_query.get("study_type", ""),
            query=query,
            abstract=abstract,
        )

        output = call_chat_model(prompt, temperature=0.0)
        decision, reason, confidence = _parse_screen_output(output)

        if not output or "Decision:" not in str(output):
            decision, reason, confidence = _fallback_screen(query, abstract)

        item = {
            **doc,
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
        }
        all_results.append(item)

        if decision == "Include":
            screened.append(item)
        else:
            excluded.append(item)

    # ── 反思轮：让LLM重新审视被排除的文献 ──
    rescued = 0
    for item in excluded:
        abstract = item.get("abstract", "")
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            population=pico_query.get("population", ""),
            intervention=pico_query.get("intervention", ""),
            comparison=pico_query.get("comparison", ""),
            outcome=pico_query.get("outcome", ""),
            study_type=pico_query.get("study_type", ""),
            query=query,
            abstract=abstract,
            reason=item.get("reason", ""),
        )

        output = call_chat_model(prompt, temperature=0.0)
        decision, reason, confidence = _parse_screen_output(output)

        if not output or "Decision:" not in str(output):
            decision, reason, confidence = _fallback_screen(query, abstract)

        if decision == "Include":
            item["decision"] = "Include"
            item["reason"] = f"【反思后捞回】{reason}"
            item["confidence"] = confidence
            screened.append(item)
            rescued += 1

    if rescued:
        print(f"反思轮捞回 {rescued} 篇文献")

    return build_state_patch(
        reranked_abstracts=all_results,
        screened_papers=screened,
        error="",
    )
