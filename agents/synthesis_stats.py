"""Layer 1: Precompute all counts, classifications, and GRADE pre-scores.

This module takes extracted PICOS + quantitative outcomes and produces a
structured stats dict. All arithmetic and classification is done in Python
(not LLM), eliminating counting errors and ensuring mutual exclusivity.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# Study type classification (mutually exclusive)
# ═══════════════════════════════════════════════════════════════

_STUDY_TYPE_RULES: List[Tuple[str, str]] = [
    # (keyword match, canonical label) — checked in order, first match wins
    ("meta-analysis", "Meta分析/系统综述"),
    ("systematic review", "Meta分析/系统综述"),
    ("meta analysis", "Meta分析/系统综述"),
    ("mendelian randomization", "孟德尔随机化"),
    ("randomized controlled", "随机对照试验(RCT)"),
    ("randomised controlled", "随机对照试验(RCT)"),
    ("rct", "随机对照试验(RCT)"),
    ("cohort", "队列研究"),
    ("prospective", "队列研究"),
    ("case-control", "病例对照研究"),
    ("case control", "病例对照研究"),
    ("cross-sectional", "横断面研究"),
    ("cross sectional", "横断面研究"),
    ("genome-wide", "全基因组关联研究(GWAS)"),
    ("gwas", "全基因组关联研究(GWAS)"),
    ("candidate gene", "候选基因关联研究"),
    ("nested case", "巢式病例对照研究"),
]


def _classify_study_type(raw: str) -> str:
    """Classify a study_type string into exactly one canonical category."""
    lowered = raw.lower().strip()
    for keyword, label in _STUDY_TYPE_RULES:
        if keyword in lowered:
            return label
    return f"其他研究类型(原文: {raw[:40]})"


# ═══════════════════════════════════════════════════════════════
# PMID-based link helpers
# ═══════════════════════════════════════════════════════════════

def _pmid_set(items: List[Dict]) -> set:
    return {str(item.get("pmid", "")) for item in items if item.get("pmid")}


# ═══════════════════════════════════════════════════════════════
# Main stats computer
# ═══════════════════════════════════════════════════════════════

def compute_stats(
    extracted_picos: List[Dict],
    quantitative_outcomes: List[Dict],
    screened_papers: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Precompute all literature stats and GRADE pre-assessments.

    Returns a dict designed for direct formatting into the synthesis prompt.
    Every number in the returned dict is computed by Python, not LLM.
    """
    n = len(extracted_picos)
    screened = screened_papers or []
    quant = quantitative_outcomes or []

    # ── 1. Study type distribution (mutually exclusive) ──
    type_counter: Counter = Counter()
    pmid_to_type: Dict[str, str] = {}
    type_pmids: Dict[str, List[str]] = {}
    for p in extracted_picos:
        pmid = str(p.get("pmid", ""))
        label = _classify_study_type(p.get("study_type", ""))
        type_counter[label] += 1
        pmid_to_type[pmid] = label
        type_pmids.setdefault(label, []).append(pmid)

    type_lines = []
    for label, count in type_counter.most_common():
        pmids = type_pmids.get(label, [])
        pmid_str = ", ".join(pmids[:8])
        if len(pmids) > 8:
            pmid_str += f" …(共{len(pmids)}篇)"
        type_lines.append(f"- {label}: **{count}** 篇 ({pmid_str})")

    # ── 2. Consistency checks ──
    checks = []
    total_by_type = sum(type_counter.values())
    if total_by_type != n:
        checks.append(f"❌ 类型分布总计 {total_by_type} ≠ 纳入总数 {n}")
    else:
        checks.append(f"✅ 类型分布总计 {total_by_type} == 纳入总数 {n}")

    if len(pmid_to_type) != n:
        checks.append(f"❌ 去重后PMID数 {len(pmid_to_type)} ≠ 纳入总数 {n}（存在重复PMID）")
    else:
        checks.append(f"✅ PMID唯一性: {len(pmid_to_type)} 篇无重复")

    # ── 3. Meta-analysis / original study overlap ──
    meta_pmids = set()
    for label, pmids in type_pmids.items():
        if "Meta" in label:
            meta_pmids.update(pmids)

    non_meta_pmids = set(pmid_to_type.keys()) - meta_pmids
    overlap_warning = ""
    if meta_pmids and non_meta_pmids:
        # We can't truly detect overlap without checking references,
        # but we flag the risk
        overlap_warning = (
            f"⚠️ 纳入 {len(meta_pmids)} 篇Meta分析 + {len(non_meta_pmids)} 篇原始研究，"
            f"存在Meta分析已纳入部分原始研究的可能性。建议人工核对Meta分析的参考文献列表，"
            f"避免数据重复使用。"
        )

    # ── 4. Effect direction / magnitude stats ──
    effect_sizes: List[float] = []
    effect_directions: List[str] = []
    ci_widths: List[float] = []
    sample_sizes: List[int] = []
    genetic_models: Counter = Counter()

    for q in quant:
        es = q.get("effect_size")
        lo = q.get("ci_lower")
        hi = q.get("ci_upper")
        if es is not None:
            effect_sizes.append(float(es))
            if lo is not None and hi is not None:
                ci_widths.append(float(hi) - float(lo))
        if lo is not None and hi is not None:
            if float(lo) > 1.0:
                effect_directions.append("有害/风险增加")
            elif float(hi) < 1.0:
                effect_directions.append("保护/风险降低")
            else:
                effect_directions.append("无统计学显著性(CI跨1)")

        tn = q.get("treatment_n") or q.get("treatment_total")
        cn = q.get("control_n") or q.get("control_total")
        if tn is not None and cn is not None:
            sample_sizes.append(int(tn) + int(cn))

        gm = q.get("genetic_model", "")
        if gm:
            genetic_models[gm] += 1

    dir_counter = Counter(effect_directions)

    effect_lines = []
    if effect_sizes:
        effect_lines.append(f"- 效应估计值数量: {len(effect_sizes)}")
        effect_lines.append(f"- 效应量范围: {min(effect_sizes):.2f} ~ {max(effect_sizes):.2f}")
        effect_lines.append(f"- 效应量中位数: {sorted(effect_sizes)[len(effect_sizes)//2]:.2f}")
    effect_lines.append(f"- 效应方向分布:")
    for d, c in dir_counter.most_common():
        effect_lines.append(f"  - {d}: {c} 项 ({c/len(effect_directions)*100:.0f}%)" if effect_directions else f"  - {d}: {c} 项")

    if genetic_models:
        effect_lines.append(f"- 遗传模型分布:")
        for m, c in genetic_models.most_common():
            effect_lines.append(f"  - {m}: {c} 项")

    # ── 5. Sample size stats ──
    sample_lines = []
    if sample_sizes:
        sample_sizes.sort()
        sample_lines.append(f"- 总样本量: {sum(sample_sizes):,}")
        sample_lines.append(f"- 中位样本量: {sample_sizes[len(sample_sizes)//2]:,}")
        sample_lines.append(f"- 样本量范围: {sample_sizes[0]:,} ~ {sample_sizes[-1]:,}")
        small = sum(1 for s in sample_sizes if s < 200)
        large = sum(1 for s in sample_sizes if s >= 2000)
        sample_lines.append(f"- 小样本研究(<200): {small} 项")
        sample_lines.append(f"- 大样本研究(≥2000): {large} 项")

    # ── 6. GRADE pre-assessment ──
    grade_pre = _pre_assess_grade(n, type_counter, effect_sizes, ci_widths, sample_sizes, dir_counter)

    # ── 7. Population / outcome / intervention summary ──
    pop_counter = Counter()
    out_counter = Counter()
    int_counter = Counter()
    for p in extracted_picos:
        pop = p.get("population", "").strip()
        out = p.get("outcome", "").strip()
        inter = p.get("intervention", "").strip()
        if pop:
            pop_counter[pop[:80]] += 1
        if out:
            out_counter[out[:80]] += 1
        if inter:
            int_counter[inter[:80]] += 1

    outcome_lines = [f"- {o}: {c} 篇" for o, c in out_counter.most_common()]
    population_lines = [f"- {p}: {c} 篇" for p, c in pop_counter.most_common()]

    # ── 8. Build result dict ──
    return {
        "n_studies": n,
        "study_type_distribution": "\n".join(type_lines),
        "consistency_checks": "\n".join(checks),
        "overlap_warning": overlap_warning,
        "effect_direction_stats": "\n".join(effect_lines) if effect_lines else "（无数值结局数据）",
        "sample_size_stats": "\n".join(sample_lines) if sample_lines else "（无样本量数据）",
        "outcome_summary": "\n".join(outcome_lines) if outcome_lines else "（无结局数据）",
        "population_summary": "\n".join(population_lines) if population_lines else "（无人群数据）",
        "grade_pre_assessment": grade_pre,
        "genetic_models_summary": (
            "\n".join(f"- {m}: {c} 项" for m, c in genetic_models.most_common())
            if genetic_models else "（未报告遗传模型）"
        ),
    }


def _pre_assess_grade(
    n: int,
    type_counter: Counter,
    effect_sizes: List[float],
    ci_widths: List[float],
    sample_sizes: List[int],
    dir_counter: Counter,
) -> str:
    """Pre-assess GRADE dimensions from observable data.

    Returns observations to GUIDE the LLM's holistic judgment — NOT mechanical
    rules to stack. The LLM must synthesize these signals into a final rating,
    not count downgrade points.
    """
    lines = ["### GRADE 预评估观测数据（供LLM综合判断，非机械规则）", ""]

    # ── Starting level ──
    has_meta = any("Meta" in t for t in type_counter)
    case_control_count = sum(c for t, c in type_counter.items() if "病例对照" in t)
    cross_sectional_count = sum(c for t, c in type_counter.items() if "横断面" in t)
    observational_count = sum(type_counter.values())

    if has_meta:
        lines.append(f"- 起始等级: Meta分析的起始等级由其纳入的原始研究类型决定。若纳入观察性研究则为**低质量(⊕⊕◯◯)**，若纳入RCT则为**高质量(⊕⊕⊕⊕)**。Meta分析本身不是独立的研究设计类型。")
    else:
        lines.append(f"- 起始等级: **低质量(⊕⊕◯◯)** — 共 {observational_count} 篇观察性研究（病例对照 {case_control_count} 篇 + 横断面 {cross_sectional_count} 篇）。按GRADE规范，观察性研究起始于低质量。")

    # ── Risk of bias observations ──
    if case_control_count >= observational_count * 0.5:
        lines.append(f"- 偏倚风险观测: 多数为病例对照设计({case_control_count}/{observational_count})。遗传关联研究重点检查——人群分层控制(PCA/基因组控制)、HWE检验、基因分型质控(call rate)、多重检验校正。")
    else:
        lines.append(f"- 偏倚风险观测: 研究设计多样性。关注选择偏倚、信息偏倚、混杂控制。")

    # ── Inconsistency observations ──
    if dir_counter:
        n_pos = dir_counter.get("有害/风险增加", 0)
        n_neg = dir_counter.get("保护/风险降低", 0)
        n_null = dir_counter.get("无统计学显著性(CI跨1)", 0)
        total_dir = n_pos + n_neg + n_null
        if total_dir > 0:
            if n_pos > 0 and n_neg > 0:
                lines.append(f"- 不一致性观测: ⚠️ 效应方向不一致——风险增加 {n_pos} 项 vs 风险降低 {n_neg} 项。方向相反提示需关注异质性来源（种族差异? 遗传模型不同? 基因分型方法差异?）。")
            elif n_pos > total_dir * 0.75:
                lines.append(f"- 不一致性观测: 效应方向一致(≥75%显示风险增加)。GRADE核心原则：即使I²较高，若方向一致可不降级。")
            else:
                lines.append(f"- 不一致性观测: 风险增加 {n_pos}/{total_dir}、无显著 {n_null}/{total_dir}。不一致性判断须基于所有证据，不应因个别小样本研究的方向不同而过度降级。")

    # ── Imprecision observations ──
    if ci_widths:
        wide_ci = sum(1 for w in ci_widths if w > 1.5)
        if wide_ci > len(ci_widths) * 0.5:
            lines.append(f"- 不精确性观测: ⚠️ {wide_ci}/{len(ci_widths)} 项CI宽度>1.5。若CI跨越无效线(OR=1)且样本量不足，提示效应估计不精确。")

    if sample_sizes:
        total_n = sum(sample_sizes)
        if total_n < 1000:
            lines.append(f"- 不精确性观测: 总样本量 {total_n:,} < 1,000（低于最优信息样本量OIS标准）。小样本阳性研究需警惕\"赢家诅咒\"效应。")
        elif total_n < 5000:
            lines.append(f"- 不精确性观测: 总样本量 {total_n:,}（中等规模）。需结合CI范围和事件数综合判断。")

    # ── Publication bias observations ──
    if n < 10:
        lines.append(f"- 发表偏倚观测: 纳入仅 {n} 篇研究，漏斗图无效（需≥10项）。注意：发表偏倚的判断应基于文献本身证据，报告的检索局限性应单独声明。")
    else:
        lines.append(f"- 发表偏倚观测: 纳入 {n} 篇研究，可使用漏斗图+Egger检验评估。遗传关联研究中发表偏倚尤其普遍（小样本阳性结果更易发表）。")

    # ── Upgrade potential observations ──
    if effect_sizes:
        large_effects = sum(1 for e in effect_sizes if e > 2.0 or e < 0.5)
        if large_effects > 0:
            lines.append(f"- 升级因素观测: {large_effects} 项效应量>2或<0.5。若统计显著、方法学质量较高、且混杂因素无法合理解释，可考虑大效应量升级(+1)。升级仅适用于观察性研究。")
        very_large = sum(1 for e in effect_sizes if e > 5.0 or e < 0.2)
        if very_large > 0:
            lines.append(f"- 升级因素观测: {very_large} 项效应量>5或<0.2，满足极大效应量标准(可考虑+2)。但需在方法学质量较高的研究中一致出现。")

    if n >= 3 and dir_counter.get("有害/风险增加", 0) >= n * 0.7:
        lines.append(f"- 升级因素观测: 多个独立研究效应方向一致(≥70%)。若来自不同种族人群且排除数据重叠，可支持剂量-反应关系或残余混杂方向性效应的升级论证。")

    # ── Core principles (from GRADE Book 2024) ──
    lines.append("")
    lines.append("### GRADE 核心原则（必须遵守，来源：GRADE Book 2024）")
    lines.append("")
    lines.append("**1. 综合判断，非机械叠加（最关键原则）**")
    lines.append("各域降级判断与最终等级之间不存在一一对应关系。5个域各降1级 ≠ 降5级。应对每个域独立判断（不严重/严重/非常严重），再综合考量各域对证据整体可信度的实质影响，得出最终等级。")
    lines.append("")
    lines.append("**2. 降级三档制**")
    lines.append("每个域: 不严重(-0) / 严重(-1) / 非常严重(-2)。必须区分严重程度，不能全部标\"严重\"。")
    lines.append("")
    lines.append("**3. 研究设计决定起始等级**")
    lines.append("观察性研究(队列/病例对照/横断面)起始: **低质量(⊕⊕◯◯)**。RCT起始: 高质量(⊕⊕⊕⊕)。Meta分析起始等级取决于纳入的原始研究类型，Meta分析本身不是独立的研究设计类型。")
    lines.append("")
    lines.append("**4. 结局导向**")
    lines.append("GRADE评级以具体结局为单位，不得对\"整个研究领域\"笼统评级。最终结论基于关键结局的评级结果。")
    lines.append("")
    lines.append("**5. 结论用语与等级严格匹配**")
    lines.append("极低质量: 只能说\"任何效应估计都非常不确定\"，禁止使用\"研究证明\"\"显著相关\"\"无关联\"等表述。**证据不足 ≠ 无关联**。")
    lines.append("")
    lines.append("**6. 升级因素仅适用于观察性研究，且使用需谨慎**")
    lines.append("已降级的证据通常不应同时升级。不应为维持某一结论而主动寻找升级理由。")
    lines.append("")
    lines.append("**7. 区分报告局限性 vs 证据质量**")
    lines.append("报告自身的局限性（如\"仅检索PubMed\"）应在评级局限性声明中单独说明，不应混入发表偏倚降级的理由。发表偏倚降级需基于文献本身的证据。")
    lines.append("")
    lines.append("**8. 禁止升降并用掩盖问题**")
    lines.append("证据因偏倚风险严重而降级的同时，又因效应量大而升级以维持原有等级——这是错误的。两者方向矛盾时需特别解释。")

    return "\n".join(lines)
