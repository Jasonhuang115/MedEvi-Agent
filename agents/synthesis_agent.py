"""Synthesis Agent: Generate GRADE evidence report with RAG + precomputed stats.

Architecture:
  Layer 1 (synthesis_stats.py) → Python precomputes all counts, classifications,
      effect stats, and GRADE pre-assessments. LLM never does arithmetic.
  Layer 2 (prompt)              → Hard constraints + RAG-retrieved guidelines
      enforce GRADE rules as non-negotiable boundaries.
  Layer 3 (LLM via prompt)      → LLM interprets precomputed stats within
      constraints and writes a fluent report.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

from agents.common import build_state_patch, state_get, to_dict
from agents.llm_router import call_chat_model
from agents.synthesis_stats import compute_stats
from prompts.synthesis_prompt import FALLBACK_SYNTHESIS_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE


def _summarise_study_types(extracted: List[Dict]) -> str:
    types = Counter(p.get("study_type", "未分类") for p in extracted)
    lines = [f"- {t}: {c} 篇" for t, c in types.most_common()]
    return "\n".join(lines) if lines else "- 暂无分类信息"


def _build_retrieval_query(extracted: List[Dict], stats: Dict[str, Any]) -> str:
    """Build a retrieval query from the precomputed stats for relevance."""
    parts = []

    type_dist = stats.get("study_type_distribution", "")
    if "病例对照" in type_dist:
        parts.append("遗传关联研究 case-control 偏倚风险 观察性研究升级")
    elif "队列" in type_dist:
        parts.append("队列研究 GRADE 偏倚风险评估")
    else:
        parts.append("观察性研究 GRADE 证据质量评估")

    n = stats.get("n_studies", len(extracted))
    if n <= 5:
        parts.append("不精确性 小样本 降级")
    if n < 10:
        parts.append("发表偏倚 单一数据库 小样本")

    parts.append("GRADE 证据总体质量 起始等级 升降级 不一致性")

    return " ".join(parts)


def _format_guidelines(guidelines: List[Dict]) -> str:
    if not guidelines:
        return "（未检索到相关规范条文，请根据GRADE通用原则判断）"

    lines = []
    for i, g in enumerate(guidelines, 1):
        lines.append(f"【规范{i}】{g['title']}")
        lines.append(g["content"])
        lines.append("")
    return "\n".join(lines)


def _format_precomputed_stats(stats: Dict[str, Any]) -> str:
    """Format the precomputed stats dict into a structured prompt section."""
    sections = []

    # Consistency checks (most important — proves Python, not LLM, did the counting)
    sections.append("### 一致性校验")
    sections.append(stats.get("consistency_checks", "（无）"))
    sections.append("")

    # Study type distribution
    sections.append(f"### 纳入研究数量: {stats.get('n_studies', 0)} 篇")
    sections.append("")
    sections.append("### 研究类型分布（每篇文献唯一分类，已互斥）")
    sections.append(stats.get("study_type_distribution", "（无）"))
    sections.append("")

    # Overlap warning
    overlap = stats.get("overlap_warning", "")
    if overlap:
        sections.append(f"### Meta分析/原始研究重叠风险")
        sections.append(overlap)
        sections.append("")

    # Population
    sections.append("### 人群汇总")
    sections.append(stats.get("population_summary", "（无）"))
    sections.append("")

    # Outcomes
    sections.append("### 结局指标汇总")
    sections.append(stats.get("outcome_summary", "（无）"))
    sections.append("")

    # Effect stats
    sections.append("### 效应量与方向统计")
    sections.append(stats.get("effect_direction_stats", "（无）"))
    sections.append("")

    # Sample size
    sections.append("### 样本量统计")
    sections.append(stats.get("sample_size_stats", "（无）"))
    sections.append("")

    # Genetic models
    sections.append("### 遗传模型分布")
    sections.append(stats.get("genetic_models_summary", "（无）"))
    sections.append("")

    # GRADE pre-assessment
    sections.append(stats.get("grade_pre_assessment", ""))

    return "\n".join(sections)


def synthesis_agent(state: Any) -> Dict[str, Any]:
    extracted: List[Dict] = to_dict(state_get(state, "extracted_picos", []) or [])
    quant: List[Dict] = to_dict(state_get(state, "quantitative_outcomes", []) or [])
    screened: List[Dict] = to_dict(state_get(state, "screened_papers", []) or [])

    if not extracted:
        return build_state_patch(grade_report="无可用纳入文献，无法生成证据报告。")

    # ═══════════════════════════════════════════════════════
    # Layer 1: Python precomputes ALL counts and stats
    # ═══════════════════════════════════════════════════════
    stats = compute_stats(extracted, quant, screened)
    precomputed_text = _format_precomputed_stats(stats)
    print(f"Stats: {stats['n_studies']} studies, checks: {stats.get('consistency_checks','')[:80]}")

    # ═══════════════════════════════════════════════════════
    # RAG: retrieve relevant GRADE/Cochrane guidelines
    # ═══════════════════════════════════════════════════════
    try:
        from tools.guideline_retriever import retrieve_guidelines

        retrieval_query = _build_retrieval_query(extracted, stats)
        guidelines = retrieve_guidelines(retrieval_query, top_k=4)
        print(f"RAG query: {retrieval_query[:100]}...")
        print(f"Retrieved {len(guidelines)} chunks: {[g['title'] for g in guidelines]}")
    except Exception as e:
        print(f"Guideline retrieval failed: {e}")
        guidelines = []

    guidelines_text = _format_guidelines(guidelines)

    # ═══════════════════════════════════════════════════════
    # Layer 2+3: Prompt with hard constraints + LLM writing
    # ═══════════════════════════════════════════════════════
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        precomputed_stats=precomputed_text,
        guidelines=guidelines_text,
        extracted_picos_list=json.dumps(extracted, ensure_ascii=False, indent=2),
    )
    report = call_chat_model(prompt, temperature=0.2)

    if not report:
        report = FALLBACK_SYNTHESIS_TEMPLATE.format(
            n_studies=len(extracted),
            study_type_summary=_summarise_study_types(extracted),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    return build_state_patch(grade_report=report, error="")
