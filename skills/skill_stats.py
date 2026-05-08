"""Skill: 统计量预计算。直接复用 synthesis_stats.compute_stats()。"""


def compute_stats(extracted: list[dict], quantitative: list[dict],
                  screened: list[dict]) -> dict:
    """预计算全量统计量和 GRADE 预评估。

    所有数字由 Python 计算，LLM 不做算术。
    """
    from agents.synthesis_stats import compute_stats as _compute

    return _compute(extracted, quantitative, screened)
