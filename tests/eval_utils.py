"""评测指标计算函数，每个函数独立可测，不依赖 LLM 或管线。"""

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
)


# ── 检索指标 ──
def calc_retrieval(retrieved_pmids, relevant_pmids, k=20):
    """输入检索结果和 ground truth，输出 Recall@k、Precision@k 和 MRR"""
    retrieved_k = retrieved_pmids[:k]
    relevant_set = set(relevant_pmids)

    hits = [1 if p in relevant_set else 0 for p in retrieved_k]
    retrieved_rel = sum(hits)

    mrr = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            mrr = 1.0 / i
            break

    return {
        f"recall@{k}": round(retrieved_rel / len(relevant_set), 4) if relevant_set else 0,
        f"precision@{k}": round(retrieved_rel / k, 4) if k else 0,
        "mrr": round(mrr, 4),
        "retrieved_total": len(retrieved_pmids),
        "relevant_total": len(relevant_set),
    }


# ── 筛选指标 ──
def calc_screening(y_true, y_pred):
    """y_true 和 y_pred 都是 list，值为 'Include' 或 'Exclude'"""
    return {
        "sensitivity": round(recall_score(y_true, y_pred, pos_label="Include", zero_division=0), 4),
        "specificity": round(recall_score(y_true, y_pred, pos_label="Exclude", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label="Include", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label="Include", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=["Include", "Exclude"]
        ).tolist(),
    }


# ── 提取指标 ──
def calc_extraction_field_accuracy(predictions, ground_truths):
    """逐字段比较，返回字段级匹配率"""
    fields = ["Population", "Intervention", "Comparison", "Outcome", "Study_Type"]
    results = {}
    for field in fields:
        matches = 0
        total = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_val = (pred.get(field, "") or "").strip()
            gt_val = (gt.get(field, "") or "").strip()
            if gt_val:
                total += 1
                if pred_val and _semantic_match(pred_val, gt_val):
                    matches += 1
        results[field] = round(matches / total, 4) if total > 0 else None
    return results


def calc_numerical_deviation(predictions, ground_truths):
    """计算数值字段的平均绝对偏差"""
    fields = ["effect_size", "ci_lower", "ci_upper", "treatment_n", "control_n"]
    deviations = {}
    for field in fields:
        diffs = []
        for pred, gt in zip(predictions, ground_truths):
            pv = pred.get(field)
            gv = gt.get(field)
            if pv is not None and gv is not None:
                diffs.append(abs(float(pv) - float(gv)))
        deviations[field] = round(sum(diffs) / len(diffs), 4) if diffs else None
    return deviations


def _semantic_match(a, b, threshold=0.70):
    """简单的语义匹配：完全相同返回 True，否则检查关键词重叠"""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return True
    a_words = set(a_lower.split())
    b_words = set(b_lower.split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap > threshold


# ── 评级指标 ──
def calc_rating_agreement(agent_ratings, expert_ratings):
    """计算总体和逐域 Kappa"""
    levels = ["high", "moderate", "low", "very_low"]
    overall_agent = [r["overall"] for r in agent_ratings]
    overall_expert = [r["overall"] for r in expert_ratings]
    overall_kappa = round(cohen_kappa_score(overall_agent, overall_expert), 4)

    domains = [
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ]
    domain_kappas = {}
    for d in domains:
        a_vals = [r.get(d, 0) for r in agent_ratings]
        e_vals = [r.get(d, 0) for r in expert_ratings]
        try:
            domain_kappas[d] = round(cohen_kappa_score(a_vals, e_vals), 4)
        except ValueError:
            domain_kappas[d] = None

    bias_dist = {"agent_higher": 0, "agent_lower": 0, "same": 0}
    for a, e in zip(overall_agent, overall_expert):
        a_idx = levels.index(a) if a in levels else -1
        e_idx = levels.index(e) if e in levels else -1
        if a_idx < e_idx:
            bias_dist["agent_higher"] += 1
        elif a_idx > e_idx:
            bias_dist["agent_lower"] += 1
        else:
            bias_dist["same"] += 1

    return {
        "overall_kappa": overall_kappa,
        "domain_kappas": domain_kappas,
        "bias_distribution": bias_dist,
    }
