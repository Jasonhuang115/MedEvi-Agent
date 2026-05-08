"""Screen Agent 评测：对 benchmark 中每条摘要调用 LLM 筛选，与 ground truth 对比。"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.eval_utils import calc_screening


def _build_picos(snp, disease):
    """从 SNP + 疾病构造默认 PICOS（与 test_10_queries.py 一致）"""
    return {
        "population": f"{disease} patients",
        "intervention": f"{snp} variant carriers",
        "comparison": "wild-type homozygous",
        "outcome": f"{disease} risk",
        "study_type": "case-control study",
    }


def screen_single_paper(title, abstract, query, pico_query):
    """调用 LLM 对单篇摘要做 Include/Exclude 判断。

    返回 (decision, reason, confidence)。
    与 agents/screen_agent.py 保持相同逻辑：LLM 筛选 + 反思轮捞回。
    """
    from agents.llm_router import call_chat_model
    from agents.screen_agent import _parse_screen_output, _fallback_screen
    from prompts.screen_prompt import SCREEN_PROMPT_TEMPLATE, REFLECTION_PROMPT_TEMPLATE

    # ── 第一轮筛选 ──
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

    # ── 反思轮（仅对首轮排除的文献） ──
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
        reflection_output = call_chat_model(reflection_prompt, temperature=0.0)
        r_decision, r_reason, r_confidence = _parse_screen_output(reflection_output)

        if not reflection_output or "Decision:" not in str(reflection_output):
            r_decision, r_reason, r_confidence = _fallback_screen(query, abstract)

        if r_decision == "Include":
            decision = "Include"
            reason = f"【反思后捞回】{r_reason}"
            confidence = r_confidence

    return decision, reason, confidence


def run_screen_eval(data_path=None):
    """跑 Screen Agent 评测，返回 (metrics, failures)"""
    if data_path is None:
        data_path = Path(__file__).parent / "benchmark_data" / "screen_benchmark.json"

    with open(data_path) as f:
        bench = json.load(f)

    samples = bench["samples"]
    y_true = []
    y_pred = []
    failures = []

    total = len(samples)
    for i, s in enumerate(samples, 1):
        snp = s["query_snp"]
        disease = s["query_disease"]
        query = f"{snp} polymorphism and {disease} risk"
        pico = _build_picos(snp, disease)

        decision, reason, confidence = screen_single_paper(
            title=s["title"],
            abstract=s["abstract"],
            query=query,
            pico_query=pico,
        )

        gt = s["ground_truth"]
        y_true.append(gt)
        y_pred.append(decision)

        status = "✅" if decision == gt else "❌"
        print(f"[{i}/{total}] {status} PMID {s['pmid']} | GT={gt} | Agent={decision} | conf={confidence:.2f}")

        if decision != gt:
            failures.append({
                "abstract_id": s["abstract_id"],
                "pmid": s["pmid"],
                "title": s["title"],
                "ground_truth": gt,
                "agent_decision": decision,
                "agent_reason": reason,
            })

    # 计算指标
    metrics = calc_screening(y_true, y_pred)
    metrics["total"] = total
    metrics["failures"] = len(failures)

    # 输出结果
    print(f"\n{'=' * 50}")
    print(f"Screen Agent Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:     {metrics['total']}")
    print(f"Sensitivity: {metrics['sensitivity']:.2%}  (基线 >0.85)")
    print(f"Specificity:  {metrics['specificity']:.2%}")
    print(f"F1:           {metrics['f1']:.2%}  (基线 >0.70)")
    print(f"Precision:    {metrics['precision']:.2%}")
    print(f"Confusion:    {metrics['confusion_matrix']}")
    print(f"Failures:     {metrics['failures']}")

    if metrics["sensitivity"] < 0.85:
        print(f"  ⚠️  Sensitivity < 0.85 baseline!")
    if metrics["f1"] < 0.70:
        print(f"  ⚠️  F1 < 0.70 baseline!")

    # 列出失败 case
    if failures:
        print(f"\n--- Failure Cases ---")
        for f in failures:
            print(f"  [{f['abstract_id']}]")
            print(f"    GT={f['ground_truth']} | Agent={f['agent_decision']}")
            print(f"    Reason: {f['agent_reason'][:120]}")

    return metrics, failures


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_screen_eval(data_path)
