"""Skill: 单篇/批量 PICOS 文献筛选（含反思轮）。独立于 LangGraph state。"""


def screen_single(title: str, abstract: str, pico_query: dict, query: str) -> dict:
    """对单篇摘要做 Include/Exclude 判断（含反思轮）。

    Returns:
        {decision: "Include"|"Exclude", reason: str, confidence: float}
    """
    from agents.llm_router import call_chat_model
    from agents.screen_agent import _parse_screen_output, _fallback_screen
    from prompts.screen_prompt import SCREEN_PROMPT_TEMPLATE, REFLECTION_PROMPT_TEMPLATE

    # 第一轮筛选
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

    # 反思轮：仅对首轮排除的文献
    if decision == "Exclude":
        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            population=pico_query.get("population", ""),
            intervention=pico_query.get("intervention", ""),
            comparison=pico_query.get("comparison", ""),
            outcome=pico_query.get("outcome", ""),
            study_type=pico_query.get("study_type", ""),
            query=query,
            abstract=abstract,
            reason=reason,
        )
        r_output = call_chat_model(reflection_prompt, temperature=0.0)
        r_decision, r_reason, r_confidence = _parse_screen_output(r_output)

        if not r_output or "Decision:" not in str(r_output):
            r_decision, r_reason, r_confidence = _fallback_screen(query, abstract)

        if r_decision == "Include":
            decision = "Include"
            reason = f"[反思后捞回] {r_reason}"
            confidence = r_confidence

    return {"decision": decision, "reason": reason, "confidence": confidence}


def screen(papers: list[dict], pico_query: dict, query: str) -> tuple[list[dict], list[dict]]:
    """批量筛选文献。

    Returns:
        (included, excluded) — 各为 [{pmid, title, abstract, decision, reason, confidence}]
    """
    included = []
    excluded = []

    for paper in papers:
        result = screen_single(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            pico_query=pico_query,
            query=query,
        )
        item = {**paper, **result}
        if result["decision"] == "Include":
            included.append(item)
        else:
            excluded.append(item)

    return included, excluded
