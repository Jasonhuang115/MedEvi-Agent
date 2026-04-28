"""Screen Agent prompts."""

SCREEN_PROMPT_TEMPLATE = """
你是一个临床文献筛选专家。请根据以下PICOS标准评估摘要是否纳入。

注意：你是宽松筛选（broad screening），不是严格排除。只要文献与PICOS主题相关，就应该纳入。
后续环节会做精细判断，本环节只排除明显不相关的文献。

Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}
Study Type: {study_type}

检索问题: {query}

摘要：
{abstract}

宽松纳排原则：
- 研究类型：case-control, cohort, cross-sectional, systematic review, meta-analysis 均可纳入。即使摘要未明确说明研究类型，只要涉及基因-表型关联分析就应纳入。
- 结局指标：只要与目标结局语义相关即可纳入。例如目标为"puberty timing"，则 precocious puberty, premature thelarche, age at menarche, pubertal development 等均视为匹配。
- 人群：不要求完全一致，相近人群即可纳入。
- 存在不确定性时，优先纳入（宁宽勿严）。

请按以下格式输出，不要添加其他内容：
Decision: [Include/Exclude]
Reason: [简短理由，若排除说明核心原因]
Confidence: [0-1]
""".strip()

REFLECTION_PROMPT_TEMPLATE = """
你在上一轮筛选中排除了以下这篇文献，现在请反思你的决策。

排除理由：{reason}

反思时请考虑：
1. 排除是因为摘要中没有明确提到，还是文献确实不符合PICOS？
2. 医学摘要常因篇幅限制省略细节，这篇文献的标题和已知信息是否暗示它可能相关？
3. 研究类型不限于case-control，systematic review和meta-analysis同样有价值。
4. 结局指标的描述可能用不同词汇表达同一概念（如precocious puberty = early puberty = 性早熟）。
5. 存在不确定性时，优先纳入（宁宽勿严），让后续环节做精细判断。

PICOS标准：
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}
Study Type: {study_type}

检索问题: {query}

摘要：
{abstract}

反思后输出新决策：
Decision: [Include/Exclude]
Reason: [反思后的理由]
Confidence: [0-1]
""".strip()
