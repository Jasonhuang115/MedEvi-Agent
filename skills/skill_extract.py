"""Skill: PICOS + 数值数据提取（含 3 层降级链）。独立于 LangGraph state。"""


def extract_picos(abstract: str) -> dict:
    """从摘要中提取 PICOS 文本字段。

    降级链: MLX LoRA → DeepSeek API → 启发式规则

    Returns:
        {population, intervention, comparison, outcome, study_type, source}
    """
    from agents.extract_agent import (
        _extract_with_mlx,
        _extract_with_llm,
        _heuristic_extract,
    )

    # Layer 1: MLX LoRA (本地)
    local = _extract_with_mlx(abstract)
    if local:
        return {**local, "source": "local"}

    # Layer 2: DeepSeek API
    llm = _extract_with_llm(abstract)
    if llm:
        return {**llm, "source": "api"}

    # Layer 3: 启发式保底
    heuristic = _heuristic_extract(abstract)
    return {**heuristic, "source": "heuristic"}


def extract_numerical(abstract: str) -> dict | None:
    """从摘要中提取数值结局数据（OR/RR/HR, 95%CI, 样本量等）。

    Returns:
        {effect_measure, effect_size, ci_lower, ci_upper, treatment_n,
         control_n, genetic_model, extraction_confidence, needs_review}
        或 None（提取失败）
    """
    from agents.extract_agent import _extract_numerical

    return _extract_numerical(abstract)
