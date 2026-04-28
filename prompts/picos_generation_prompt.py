"""Prompt for auto-generating PICOS + query from simple SNP/disease inputs."""

PICOS_GENERATION_PROMPT = """你是一个临床研究助手。用户提供了2个关键信息，请根据这些信息自动生成完整的检索问题(query)和PICOS标准。

输入信息：
- 基因/SNP：{snp}
- 疾病/表型：{disease}

请根据以上信息生成：
1. query：一段自然的英文检索问题，描述该SNP与该疾病之间的关联关系。格式如："{snp} polymorphism and {disease} risk/susceptibility"
2. population：目标人群描述（英文）
3. intervention：干预/暴露因素描述（英文），即SNP variant carriers
4. comparison：对照措施描述（英文），通常为wild-type homozygous
5. outcome：结局指标描述（英文），即疾病风险或易感性
6. study_type：根据该领域的常见研究设计推断一个最可能的研究类型（英文，如"case-control study"、"cohort study"、"cross-sectional study"）

要求：
- 基因名用正式HUGO gene symbol（如ESR1、TP53）
- rs号保持原样
- 疾病名称使用标准医学术语
- 结局指标根据疾病性质选择合适的措辞：risk（风险）、susceptibility（易感性）、prognosis（预后）等
- 如果没有SNP只有基因名，忽略rs号部分即可

请严格按照以下JSON格式输出，不要添加任何其他内容：
{{"query": "...", "population": "...", "intervention": "...", "comparison": "...", "outcome": "...", "study_type": "..."}}""".strip()
