"""Chat Agent 工具定义：OpenAI function calling schema + handler 绑定。

7 个工具：4 个统计查询 + 1 个详情查询 + 1 个规范检索 + 1 个过滤重算
"""

# OpenAI function calling 格式的 tool schemas
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_study_type_distribution",
            "description": "查看纳入文献的研究类型分布（病例对照/队列/GWAS/Meta分析等），每篇被分入唯一类型。包含每类对应的PMID列表。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_effect_stats",
            "description": "查看效应量统计：方向分布（风险增加/保护/无显著）、效应量范围与中位数、遗传模型分布",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_size_stats",
            "description": "查看样本量统计：总样本量、中位数、范围、小样本(<200)/大样本(≥2000)计数",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grade_pre_assessment",
            "description": "查看GRADE预评估：5个降级域（偏倚风险、不一致性、间接性、不精确性、发表偏倚）的观测数据和升级因素，以及核心原则",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_detail",
            "description": "查看某篇文献的详细PICOS信息和提取的数值数据（OR/RR、95%CI、样本量、遗传模型等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "PubMed ID，如 12345678",
                    }
                },
                "required": ["pmid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_guideline",
            "description": "从GRADE权威规范知识库中检索与特定话题相关的官方条文（如偏倚风险评估标准、不精确性判断标准等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "检索主题，如偏倚风险、不一致性、发表偏倚、不精确性、间接性、升级因素",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_and_recompute",
            "description": "排除指定文献（如某篇Meta分析或低质量研究）后，重新计算所有统计量和GRADE预评估。用于回答'排除某篇后结论会变吗'这类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "exclude_pmids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要排除的文献PMID列表",
                    }
                },
                "required": ["exclude_pmids"],
            },
        },
    },
]


# ── Handler 工厂（带会话级缓存） ──

_cache = {"context_id": None, "handlers": None}


def build_handlers(pipeline_context: dict):
    """绑定工具处理器到当前管线的数据上下文。

    pipeline_context: {
        'extracted_picos': [...],
        'quantitative_outcomes': [...],
        'screened_papers': [...],
        'grade_report': str,
        'query': str,
        'pico_query': dict,
    }

    缓存策略：pipeline_context 不变则 full_stats 不重算。
    """
    global _cache
    ctx_id = id(pipeline_context)
    if _cache["context_id"] == ctx_id and _cache["handlers"] is not None:
        return _cache["handlers"]

    from skills.skill_stats import compute_stats as skill_compute_stats
    from skills.skill_guideline import retrieve_guidelines as skill_guideline

    extracted = pipeline_context.get("extracted_picos", [])
    quant = pipeline_context.get("quantitative_outcomes", [])
    screened = pipeline_context.get("screened_papers", [])

    # 预计算完整统计量（一次）
    full_stats = skill_compute_stats(extracted, quant, screened)

    def _get_distribution():
        return full_stats.get("study_type_distribution", "")

    def _get_effect():
        return full_stats.get("effect_direction_stats", "")

    def _get_sample():
        return full_stats.get("sample_size_stats", "")

    def _get_grade_pre():
        return full_stats.get("grade_pre_assessment", "")

    def _get_paper_detail(pmid: str):
        for p in extracted:
            if str(p.get("pmid")) == pmid:
                num = next(
                    (q for q in quant if str(q.get("pmid")) == pmid), {}
                )
                return {
                    "pmid": pmid,
                    "picos": {
                        k: p.get(k, "")
                        for k in [
                            "population",
                            "intervention",
                            "comparison",
                            "outcome",
                            "study_type",
                        ]
                    },
                    "numerical": {
                        k: num.get(k)
                        for k in [
                            "effect_measure",
                            "effect_size",
                            "ci_lower",
                            "ci_upper",
                            "treatment_n",
                            "control_n",
                            "genetic_model",
                            "extraction_confidence",
                        ]
                    },
                }
        return {"error": f"PMID {pmid} not found in extracted papers"}

    def _retrieve(topic: str):
        guidelines = skill_guideline(topic, top_k=3)
        return [
            {"title": g["title"], "content": g["content"][:500]}
            for g in guidelines
        ]

    def _filter_recompute(exclude_pmids: list):
        filtered_extracted = [
            p for p in extracted if str(p.get("pmid")) not in exclude_pmids
        ]
        filtered_quant = [
            q for q in quant if str(q.get("pmid")) not in exclude_pmids
        ]
        filtered_screened = [
            s for s in screened if str(s.get("pmid")) not in exclude_pmids
        ]
        new_stats = skill_compute_stats(
            filtered_extracted, filtered_quant, filtered_screened
        )
        return {
            "n_after_exclusion": len(filtered_extracted),
            "n_excluded": len(extracted) - len(filtered_extracted),
            "study_type_distribution": new_stats.get(
                "study_type_distribution", ""
            ),
            "effect_direction_stats": new_stats.get(
                "effect_direction_stats", ""
            ),
            "sample_size_stats": new_stats.get("sample_size_stats", ""),
            "grade_pre_assessment": new_stats.get("grade_pre_assessment", ""),
        }

    handlers = {
        "get_study_type_distribution": _get_distribution,
        "get_effect_stats": _get_effect,
        "get_sample_size_stats": _get_sample,
        "get_grade_pre_assessment": _get_grade_pre,
        "get_paper_detail": _get_paper_detail,
        "retrieve_guideline": _retrieve,
        "filter_and_recompute": _filter_recompute,
    }

    _cache["context_id"] = ctx_id
    _cache["handlers"] = handlers
    return handlers
