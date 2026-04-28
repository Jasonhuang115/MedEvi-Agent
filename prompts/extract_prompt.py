"""Extract Agent prompts."""

EXTRACT_PROMPT_TEMPLATE = """
你是医疗实体提取专家。请从下面摘要中提取PICOS，并严格输出JSON。

输出JSON键名必须是：
Population, Intervention, Comparison, Outcome, Study_Type

摘要：
{abstract}

只输出JSON，不要输出解释。
""".strip()


NUMERICAL_EXTRACTION_PROMPT = """你是临床研究数据提取专家。从以下摘要中提取数值结局数据。

要求：
- 只提取主要结局（primary outcome）的数据
- 如果摘要报告了多个结局，只取第一个明确陈述的
- 数值必须从原文精确复制，不要计算或推断
- 如果某字段在摘要中未报告，设为 null

对于二分类结局（OR/RR/HR）：
- 提取每组的样本量和事件数
- 提取效应量和95%置信区间

对于连续型结局（MD/SMD）：
- 提取每组的均值、标准差、样本量
- 提取均数差和95%置信区间

对于遗传关联研究：
- 识别遗传模型（allelic/dominant/recessive/additive）
- 提取对应的OR和95%CI

摘要：
{abstract}

只输出JSON，格式：
{{
  "outcome_label": "主要结局描述",
  "outcome_type": "binary或continuous或time-to-event",
  "treatment_n": 数字或null,
  "treatment_events": 数字或null,
  "control_n": 数字或null,
  "control_events": 数字或null,
  "treatment_mean": 数字或null,
  "treatment_sd": 数字或null,
  "control_mean": 数字或null,
  "control_sd": 数字或null,
  "effect_measure": "OR或RR或HR或MD或SMD",
  "effect_size": 数字或null,
  "ci_lower": 数字或null,
  "ci_upper": 数字或null,
  "p_value": 数字或null,
  "genetic_model": "allelic或dominant或recessive或additive或空字符串",
  "extraction_confidence": "HIGH或MEDIUM或LOW",
  "needs_review_reason": "如为LOW，说明原因；否则空字符串"
}}""".strip()
