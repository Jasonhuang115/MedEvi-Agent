# MedEvi-Agent 项目技术总结

---

## 一、项目概述

MedEvi-Agent 是一个**临床证据自动化合成系统**，用户只需输入3个字段（SNP/基因、疾病/表型、研究类型），系统即可自动完成：PubMed文献检索 → 语义重排序 → PICOS筛选 → 结构化数据提取 → GRADE证据质量报告生成。

核心技术特征：
- **4-Agent 管线架构**（Search → Screen → Extract → Synthesis），基于 LangGraph 编排
- **Small-to-Large 混合推理**：本地 LoRA 微调模型处理高频结构化提取，云端 DeepSeek API 处理复杂推理
- **RAG 增强的证据评级**：检索 GRADE/Cochrane 权威规范条文，锚定报告结论
- **完整可观测性**：LangSmith 追踪全链路 LLM 调用，记录 Token 消耗、延迟和 Prompt 全貌

**适用场景**：遗传关联研究的快速证据概览，帮助研究者快速了解某个 SNP-疾病关联的现有文献全貌，而非替代正式的系统综述。

---

## 二、开发流程

### 2.1 整体开发路径

```
Phase 1: 管线搭建
  → LangGraph 4节点管线 + Pydantic状态模型 + Streamlit UI

Phase 2: 数据准备 + 本地模型训练
  → 10-700条摘要收集 → DeepSeek标注 → MLX LoRA微调 Qwen2.5-1.5B
  → 3轮迭代(V1过拟合→V2正则化→V3扩数据)

Phase 3: 检索质量优化
  → LLM查询扩展(自然语言→PubMed Boolean) + Cohere语义重排序
  → 宽严筛选策略(从严格排除→宽松纳入+反思捞回)

Phase 4: 证据评级增强（两轮迭代）
  → RAG注入GRADE规范 + Python预计算替代LLM计数
  → 三层架构(统计→约束→写作)消除报告错误
  → V1→V2 RAG升级：替换错误GRADE条文，引入综合判断原则+AI自查清单

Phase 5: 可观测性 + 用户体验
  → LangSmith全链路追踪 + 流式进度展示
  → 6字段→3字段输入简化 + 去除内部实现细节暴露
```

### 2.2 LoRA 微调迭代历程

| 版本 | 样本量 | LoRA配置 | 验证Loss | 结果 |
|------|--------|----------|----------|------|
| V1 | 40 | rank=32, dropout=0, lr=1e-5 | 3.53 | 严重过拟合，输出大量乱码 |
| V2 | 40 | rank=8, dropout=0.1, lr=5e-6, grad_accum=4 | — | 过拟合消失，收敛健康 |
| V3(生产) | 473 train / 84 val | rank=8, dropout=0.1, lr=5e-6, seq_len=1536 | 1.397 | 稳定泛化，精度良好 |

### 2.3 10-SNP 基准测试

10个不同SNP-疾病组合全部通过，PubMed检索6~50篇（均值25.1），筛选纳入5~20篇（均值12.8），耗时78~197秒/查询。

---

## 三、技术栈详解

### 3.1 Agent架构与编排 — LangGraph

**技术：** LangGraph `StateGraph` + Pydantic v2 类型化状态

```
  search ──▶ screen ──▶ extract ──▶ synthesis ──▶ END
    │           │
    │ 0结果     │ 0篇通过
    ▼           ▼
  synthesis   END
```

- **条件路由**：`_route_after_search()` 在0结果时跳过筛选直接生成空报告；`_route_after_screen()` 在无纳入文献时终止管线
- **节点隔离**：每个 Agent 节点被 `_safe_node()` 包装，单节点异常不影响管线继续执行
- **不可变状态**：每个节点返回 dict patch（而非直接修改状态），LangGraph 自动合并，保证状态流转的确定性
- **流式执行**：`graph_app.stream()` 逐节点 yield 中间结果，前端实时展示每个阶段的进展

**对应文件：** `graph.py`, `state.py`

---

### 3.2 本地模型部署 — MLX + LoRA

**技术：** Apple MLX 框架 + LoRA (Low-Rank Adaptation)

- **基座模型**：Qwen2.5-1.5B-Instruct（15亿参数）
- **微调方式**：LoRA rank=8，仅训练约0.3%参数，在 Apple Silicon M4 上完成
- **推理性能**：单篇摘要PICOS提取 ~1.85秒，完全本地运行，零API成本
- **模型管理**：环境变量 `MLX_PICOS_MODEL_PATH` 控制版本切换，支持热升级

**Small-to-Large 混合推理架构：**

| 任务 | 执行者 | 原因 |
|------|--------|------|
| PICOS结构化提取 | 本地 LoRA 模型 | 高频调用、格式固定、延迟敏感 |
| 文献筛选/报告生成 | DeepSeek API | 需要复杂推理、低频调用 |
| 查询扩展/PICOS生成 | DeepSeek API | 需要领域知识和同义词推理 |

**三层回退机制（Extract Agent）：**
```
MLX LoRA (首选) → DeepSeek API (回退) → 启发式规则 (兜底)
```
任一层的失败自动切换到下一层，确保管线不会因为单点故障而中断。

**对应文件：** `agents/mlx_extractor.py`, `agents/extract_agent.py`, `models/Qwen2.5-1.5B-Med-PICOS_v3/`

---

### 3.3 RAG（检索增强生成）— GRADE 规范检索与注入

**应用场景：Synthesis Agent 的证据评级**

RAG 是本项目中唯一涉及多版本迭代的模块——规范条文的质量直接决定 GRADE 报告的准确性，因此经历了 V1→V2 的实质性升级。

---

#### 3.3.1 架构总览

```
synthesis_agent.py
  ├─ Layer 1: compute_stats() → 观测数据 (synthesis_stats.py)
  ├─ Layer 2: _build_retrieval_query() → 检索词
  │             └─ retrieve_guidelines(query, top_k=4) → 4段最相关规范
  │                  └─ GuidelineRetriever (guideline_retriever.py)
  │                       └─ TF-IDF + 余弦相似度
  │                            └─ GUIDELINE_CHUNKS (guideline_store.py, 14段)
  └─ Layer 3: SYNTHESIS_PROMPT_TEMPLATE.format(guidelines=...)
                → LLM 综合判断 → GRADE 报告
```

| 组件 | 技术 | 说明 |
|------|------|------|
| 知识库 V2 | `guideline_store.py` | **14 段** GRADE 规范条文（中文），覆盖前提条件→PICO构建→起始等级→5个降级域→3个升级因素→综合判断→输出格式→结论用语→AI常见错误清单 |
| 检索器 | `guideline_retriever.py` | TF-IDF + 余弦相似度，自定义中英文字符级 bigram tokenizer |
| 查询构建 | `_build_retrieval_query()` | 根据纳入文献特征（研究类型、样本量、效应方向）自动生成检索词 |
| 上下文注入 | `synthesis_prompt.py` | 检索到的规范条文 + 预计算观测数据 + 核心原则 + 自查清单，全部注入 prompt |

---

#### 3.3.2 知识库 V1 → V2 的演进

**V1 的问题（对应原 `guideline_store.py`，10 段）：**

V1 的 GRADE 条文基于通用搜索引擎结果拼凑，存在三个影响评级结果的关键性错误：

| 问题 | V1 错误 | GRADE Book 2024 正确规定 | 影响 |
|------|---------|--------------------------|------|
| 起始等级 | 观察性研究(队列/病例对照)起始于"低质量" | 观察性研究确认为低质量起步，但需区分非随机干预研究(NRSI)可用 ROBINS-I 评估后提升 | 等级正确但缺少灵活提升路径 |
| 降级幅度 | 单维度可降 2 级（严重-1 / 非常严重-2） | 每个域最多降 1 级的主要判断；**最终等级是综合判断而非机械叠加** | V1 最大的错误是把"各域降级次数之和"等同于"总降级数" |
| 升级因素 | 列了 3 个但不完整，且无累计上限 | 3 个升级因素明确：大效应量、剂量-反应关系、残余混杂方向性效应。累计升级不超过 2 级。已降级的证据通常不应同时升级 | 可能升降互抵掩盖问题 |

此外，V1 还缺失了以下关键内容：
- **结局导向要求**：未要求按具体结局分别评级
- **AI 常见错误清单**：未内嵌自查机制
- **标准化结论用语模板**：缺少"证据不足 ≠ 无关联"等硬约束
- **Meta 分析归类规则**：Meta 分析本身不是独立研究设计类型

**V2 改进（14 段，依据 GRADE Book 2024 + JCE 2011 系列 + ACIP 2024）：**

V2 将完整的 GRADE 操作指南按语义切分为 14 个自包含的检索单元：

| # | Chunk ID | 内容 | 检索触发词 |
|---|----------|------|-----------|
| 0 | `grade_prerequisites` | 3 项前提条件（证据体要求、结局导向、PICO 先行） | 系统检索、PICO |
| 1 | `grade_pico` | PICO 构建 + 结局重要性分级（关键 7-9 / 重要 4-6 / 次要 1-3） | PICO、结局、重要性分级 |
| 2 | `grade_starting_level` | 研究设计→起始等级（含 Meta 分析归类规则） | 研究设计、起始等级、Meta |
| 3 | `grade_risk_of_bias` | 降级域 1：观察性/RCT 评估清单 + 遗传专属项 | 偏倚风险、HWE、人群分层 |
| 4 | `grade_inconsistency` | 降级域 2：效应方向+I²+亚组解释 | 不一致性、异质性、I² |
| 5 | `grade_indirectness` | 降级域 3：人群/暴露/结局三类间接性 | 间接性、替代结局、标签SNP |
| 6 | `grade_imprecision` | 降级域 4：CI 宽度+OIS+赢家诅咒 | 不精确性、置信区间、样本量 |
| 7 | `grade_publication_bias` | 降级域 5：**区分报告局限 vs 证据本身偏倚** | 发表偏倚、漏斗图、Egger |
| 8 | `grade_upgrade` | 3 个升级因素（仅观察性研究、累计≤2级、不与降级并用） | 升级、大效应量、剂量反应 |
| 9 | `grade_final_rating` | **核心：综合判断非机械叠加**（含正误示例对比） | 综合判断、最终等级、机械叠加 |
| 10 | `grade_genetic` | 遗传关联专属（HWE、PCA、多重检验、赢家诅咒、功能注释） | 遗传关联、病例对照、HWE |
| 11 | `grade_output_format` | 报告必需 6 部分结构 | 输出格式、报告结构 |
| 12 | `grade_conclusion_language` | 标准化结论用语 + 禁止用语（"证明""无关联"） | 结论用语、极低质量、不确定 |
| 13 | `grade_ai_errors` | AI 必避 8 大错误（自查清单） | 常见错误、机械叠加、过度自信 |

**Chunk 设计原则：**
1. **语义自包含** — 每个 chunk 可独立回答一个具体问题（如"偏倚风险怎么评""不精确性降级标准是什么"），不依赖上下文
2. **检索对齐** — 标题和首句包含该域的核心关键词，确保 TF-IDF 能匹配到
3. **可操作** — 每段包含评估清单 + 降级标准 + 特殊考量，而非纯理论描述
4. **中文为主** — 与 LLM prompt 和报告输出语言一致，减少翻译偏差

---

#### 3.3.3 检索器实现细节

**TF-IDF + 余弦相似度的本地检索方案：**

```python
# tools/guideline_retriever.py — 核心类
class GuidelineRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=_char_tokenizer,  # 中英文字符级 bigram
            max_features=500,
        )
        corpus = [c["title"] + " " + c["content"] for c in GUIDELINE_CHUNKS]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {**GUIDELINE_CHUNKS[i], "score": float(scores[i])}
            for i in top_indices
        ]
```

**自定义 Tokenizer 设计：**

中文无天然分词边界，英文医学术语含连字符和数字。采用"保留原词 + 中文字符 bigram"策略：

```python
def _char_tokenizer(text: str) -> List[str]:
    tokens = []
    for word in text.split():
        chinese_chars = [c for c in word if '一' <= c <= '鿿']
        if len(chinese_chars) > 2:
            tokens.append(word)                    # 保留原词
            for i in range(len(chinese_chars) - 1):
                tokens.append(chinese_chars[i] + chinese_chars[i+1])  # 字符bigram
        else:
            tokens.append(word)
    return tokens
```

这种 tokenizer 的优势：中文"发表偏倚"会被切为 `["发表偏倚", "发表", "表偏", "偏倚"]`，既保留了完整术语，又通过 bigram 提高了部分匹配的召回。

**查询构建策略：**

```python
# agents/synthesis_agent.py
def _build_retrieval_query(extracted, stats) -> str:
    """根据实际纳入文献特征，动态构建检索词。"""
    parts = []
    # 研究类型特征 → 偏倚风险 + 遗传关联
    if "病例对照" in type_dist:
        parts.append("遗传关联研究 case-control 偏倚风险 HWE检验 PCA")
    elif "队列" in type_dist:
        parts.append("队列研究 GRADE 偏倚风险评估")
    # 样本量特征 → 不精确性
    if n <= 5:
        parts.append("不精确性 小样本 降级 赢家诅咒")
    if n < 10:
        parts.append("发表偏倚 漏斗图 单一数据库 边界规则")
    # 通用
    parts.append("GRADE 最终等级 综合判断 非机械叠加 结论用语")
    return " ".join(parts)
```

**检索效果验证（5 种典型场景）：**

| 查询场景 | Top-1 命中 | Score | Top-2 命中 |
|----------|-----------|-------|-----------|
| 遗传关联 + 偏倚风险 + 不精确性 | `grade_genetic` | 0.420 | `grade_risk_of_bias` |
| 小样本 + 发表偏倚 + 单一数据库 | `grade_publication_bias` | 0.309 | `grade_genetic` |
| GRADE 总体质量 + 升降级 + 不一致性 | `grade_final_rating` | 0.354 | `grade_inconsistency` |
| 结论用语 + 标准化表述 + 极低质量 | `grade_ai_errors` | 0.294 | `grade_conclusion_language` |
| AI 常见错误 + 机械叠加 | `grade_ai_errors` | 0.335 | `grade_publication_bias` |

14 段语料、术语明确，TF-IDF 关键词匹配精度足够。零延迟、零成本、零外部依赖。语料规模从 10 段增长到 14 段仍保持在本地检索的最佳范围。

---

#### 3.3.4 RAG 注入与综合判断流程

检索到的规范条文与预计算观测数据一并注入 prompt，LLM 被要求：

1. **先读规范** — 参考 RAG 检索到的 GRADE 权威条文进行判断
2. **再看数据** — 引用 Python 预计算的观测数据（效应方向、CI 宽度、样本量）
3. **综合判断** — 对各域做独立评估后，综合考量对证据整体可信度的实质影响
4. **自查输出** — 逐项确认输出前自查清单（8 大 AI 常见错误）

这种设计使得 LLM 从"凭记忆评级"变为"参考权威规范 + 基于观测数据 + 遵循综合判断原则"的结构化推理。

**对应文件：** `tools/guideline_store.py`, `tools/guideline_retriever.py`, `agents/synthesis_agent.py`, `prompts/synthesis_prompt.py`

---

### 3.4 Prompt Engineering

**5类 Prompt 模板，涵盖完整管线：**

| Prompt | 所属阶段 | 关键技巧 |
|--------|----------|----------|
| `QUERY_EXPANSION_PROMPT` | Search | 角色分配("PubMed检索专家") + 6条改写规则 + 同义词展开示例 + 输出长度约束 |
| `SCREEN_PROMPT_TEMPLATE` | Screen | PICOS结构化评估 + **宽松偏置**("宁宽勿严") + 具体纳排规则示例 + 结构化输出(`Decision:/Reason:/Confidence:`) |
| `REFLECTION_PROMPT_TEMPLATE` | Screen(反思) | **反思/自我批判模式** — LLM重新审视被排除文献，列出5个重新考虑的维度 |
| `EXTRACT_PROMPT_TEMPLATE` | Extract | JSON-only约束 + 固定键名 + 空字段处理规则 |
| `NUMERICAL_EXTRACTION_PROMPT` | Extract | 详细字段规格 + 条件提取规则 + 精确复制约束 + 遗传模型特殊处理 |
| `SYNTHESIS_PROMPT_TEMPLATE` | Synthesis | **三层架构注入** + 硬约束列表 + 预计算数据引用要求 |
| `PICOS_GENERATION_PROMPT` | 输入预处理 | 3字段→完整PICOS扩展 + HUGO基因符号标准化 |

**关键 Prompt Engineering 技巧：**

1. **角色效应与决策边界控制**：Screen Agent 从"严格排除"改为"宽松筛选"后，纳入率从5/27提升到14/20。同一 LLM，仅改变角色定位，决策边界显著位移。

2. **显式规则注入替代隐式依赖**：在 prompt 中显式列出"precocious puberty = early puberty = 性早熟"等效关系，解决了 LLM 按字面匹配而非语义匹配的问题。

3. **反思循环 (Reflection Loop)**：被排除的文献进入第二轮评估，LLM 被告知之前排除的理由并要求重新考虑，成功捞回被错误排除的文献。

4. **硬约束与软指导分离**：Synthesis prompt 中，计数/分类由 Python 预计算（硬约束），LLM 只负责解读和表述（软任务），消除了 LLM 做算术的错误。

5. **结构化输出控制**：使用 `Decision:/Reason:/Confidence:` 等固定前缀使解析正则化，降低输出解析失败率。

**对应文件：** `prompts/search_prompt.py`, `prompts/screen_prompt.py`, `prompts/extract_prompt.py`, `prompts/synthesis_prompt.py`, `prompts/picos_generation_prompt.py`

---

### 3.5 语义检索与重排序

**Cohere Reranker (`rerank-english-v3.0`)**

- 位置：Search Agent 在 PubMed 检索后、Screening 之前
- 作用：将关键词召回的前50篇文献按语义相关性重新排序，取 Top-20 进入筛选
- 设计细节：用**原始自然语言query**（而非扩展后的Boolean query）做 rerank，因为语义匹配需要自然语言上下文
- 容错：rerank 异常时降级为取前20篇，不阻塞管线

**对应文件：** `tools/reranker.py`

---

### 3.6 可观测性 — LangSmith

**追踪范围：**
- 每次 LLM 调用的 Prompt 输入、Response 输出、延迟、Token 消耗
- 通过 `langsmith.traceable` 装饰器自动记录，零侵入性
- 条件启用：仅在 `LANGCHAIN_TRACING_V2=true` 时激活，关闭时零开销

**环境配置：**
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=MedEvi-Agent
```

**对应文件：** `agents/llm_router.py`

---

### 3.7 三层架构（Synthesis Agent）

这是项目中最能体现架构设计思想的模块：

```
Layer 1: Python 预计算层 (synthesis_stats.py)
  ├─ 研究类型互斥分类（17条关键词匹配规则，first-match-wins）
  ├─ 一致性校验（总数 == 子类之和，PMID去重验证）
  ├─ 效应方向/量级统计
  ├─ 样本量概要
  ├─ Meta分析/原始研究重叠风险检测
  └─ GRADE维度预评估（基于可观测数据）

Layer 2: 约束生成层
  ├─ RAG检索GRADE/Cochrane规范条文 (guideline_retriever.py, 14段语义chunk)
  └─ GRADE核心原则（"综合判断非机械叠加"、"结局导向"、"证据不足≠无关联"等8条）

Layer 3: LLM 写作层
  └─ DeepSeek 解读预计算数据 + 遵循硬约束 + 参考规范 → 生成流畅报告
```

**设计原则：LLM 负责"表述"，Python 负责"事实"。**

---

### 3.8 前端架构 — Streamlit

- **异步执行**：管线在 `threading.Thread` 后台运行，UI 线程每 0.5 秒轮询共享 dict 更新进度
- **流式进度**：`graph_app.stream()` 每完成一个节点，立即更新进度展示
- **3-Tab 结果展示**：检索与筛选 → PICOS提取+数值数据 → GRADE证据报告
- **状态持久化**：`st.session_state["result_state"]` 保持上次运行结果
- **CSV 导出**：筛选结果和提取数据均可下载

**对应文件：** `app.py`

---

### 3.9 状态管理 — Pydantic v2 + 不可变模式

- `PaperState(BaseModel)` 定义11个字段，类型安全
- 节点返回 `build_state_patch()` 生成的 dict patch，LangGraph 自动合并
- `state_get()` 同时兼容 dict 和 pydantic 对象访问
- 嵌套类型：`Paper`, `ExtractedPICOS`, `QuantitativeOutcome`

**对应文件：** `state.py`, `agents/common.py`

---

### 3.10 数据处理管线

- **自动化标注**：DeepSeek API 批量标注PICOS实体，支持断点续标（每50条自动保存）
- **质量过滤**：5字段非空检查，摘要哈希去重
- **训练/验证划分**：85/15 分层划分
- **MLX LoRA 训练**：YAML配置驱动，支持 grad_accumulation、seq_len 调整

**对应文件：** `scripts/expand_dataset.py`, `scripts/resume_annotate.py`

---

### 3.11 技术全景图

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│  (后台线程 + 流式进度 + Tab展示 + CSV导出)               │
├─────────────────────────────────────────────────────────┤
│                     LangGraph 管线编排                    │
│  search → screen → extract → synthesis                 │
│  条件路由 / 节点隔离 / 不可变状态 / LangSmith追踪         │
├───────────┬───────────┬────────────┬───────────────────┤
│ Search    │ Screen    │ Extract    │ Synthesis         │
│ Agent     │ Agent     │ Agent      │ Agent             │
│           │           │            │                   │
│ LLM查询扩展│ PICOS评估  │ MLX LoRA   │ Layer1: Python统计 │
│ PubMed API│ 反思捞回   │ DeepSeek   │ Layer2: RAG检索   │
│ Cohere    │ keyword   │ Heuristic  │ Layer3: LLM写作   │
│ Rerank    │ fallback  │ fallback   │ GRADE硬约束       │
├───────────┴───────────┴────────────┴───────────────────┤
│                    Prompt Engineering                   │
│  角色效应 / 宽松偏置 / 反思循环 / 硬约束 / 结构化输出     │
├─────────────────────────────────────────────────────────┤
│              本地模型: MLX + LoRA (Qwen2.5-1.5B)        │
│              远程API: DeepSeek + Cohere + PubMed        │
│              可观测性: LangSmith Tracing                 │
│              状态管理: Pydantic v2 + LangGraph State     │
└─────────────────────────────────────────────────────────┘
```

---

## 四、问题与优化

### 问题1：PubMed 关键词检索召回率低

**现象：** "ESR1 rs9340799 polymorphism and puberty" → 仅3篇结果
**根因：** PubMed 是关键词匹配搜索引擎，不做语义扩展。"puberty"不会自动匹配"pubertal"、"menarche"、"sexual maturation"
**解决：** LLM 查询扩展——将自然语言查询改写为 PubMed Boolean 查询，为基因名、rs号、表型分别添加同义词 OR 组
**效果：** 3篇 → 27篇

### 问题2：筛选过于严格，LLM做字面匹配

**现象：** "central precocious puberty" 被LLM判定为"不是 puberty timing"，Meta分析被排除因为"不是 case-control study"
**根因：** LLM 在严格排除的角色设定下按字面含义匹配，缺乏医学术语的语义泛化能力
**解决：** 双重修复——(1) 角色从"严格排除"改为"宽松筛选"，增加显式语义等效示例；(2) 增加反思轮，被排除文献二次评估
**效果：** 1/27纳入 → 14/20纳入（含2篇反思捞回）

### 问题3：LLM做计数和分类 → 报告数据错误

**现象：** 横断面研究"5篇"但列出6个PMID，PMID被重复归类，总数不对
**根因：** LLM 不擅长精确计数和集合运算，JSON dump 给 LLM 让 LLM 自己分类必然出错
**解决：** 三层架构——Python 预计算所有统计量（互斥分类、一致性校验、效应统计），LLM只负责解读和表述
**效果：** 报告中所有数字由 Python 计算，"5篇(列出6个PMID)"此类错误彻底消除

### 问题4：GRADE 规范条文不准确 → 评级系统性偏差（RAG V1→V2 升级）

这是所有问题中最根本的一个——因为 RAG 检索到的规范本身就是错的，导致 LLM 无论怎么遵循规范都会产出有偏的评级。

**V1 知识库的三个关键性错误：**
1. **降级规则被描述为"机械叠加"**：10 段指南暗示「5 域各降 1 级 = 降 5 级」，但 GRADE Book 2024 明确规定各域判断与最终等级间**不存在一一对应关系**，最终等级是综合判断的结果
2. **缺少结局导向要求**：未要求按具体结局分别评级，导致 LLM 对"整个研究领域"笼统评级
3. **缺少 AI 常见错误自查清单**：无"机械叠加""升降互抵""过度自信结论用语"等 AI 典型错误的防御机制

此外，缺少标准化结论用语模板、Meta 分析归类规则（Meta 分析本身不是独立研究设计类型）、升级因素累计上限（≤2 级）、区分报告局限 vs 证据偏倚等关键约束。

**V2 解决（4 项改进）：**
1. **权威规范源替换**：以 GRADE Book 2024 + JCE 2011 系列 + ACIP GRADE Handbook 2024 为唯一依据，完整重写 14 段语义 chunk
2. **"综合判断"原则置顶**：在 `grade_final_rating` chunk 中给出正误对比示例，明确禁止机械叠加
3. **嵌入自查清单**：新增 `grade_ai_errors` chunk（8 大 AI 常见错误）+ `grade_conclusion_language` chunk（标准化用语 + 禁止用语）
4. **prompt 层面三层防御**：(a) 核心原则前置提醒；(b) RAG 规范注入；(c) 输出前 10 项自查清单

**效果：** 不再出现"5 域各降 1 级共降 5 级"这类违反 GRADE 基本原理的错误。报告明确区分"综合判断过程"和"机械叠加"，结论用语与等级严格匹配。"证据不足 ≠ 无关联"这一关键约束确保极低质量证据不会被错误表述为"无关联"。

### 问题5：LoRA 微调过拟合

**现象：** V1 模型验证 loss 高达 3.53，输出大量重复乱码
**根因：** 仅40个样本，LoRA rank=32（过高），无 dropout 正则化
**解决：** (1) rank 32→8；(2) 添加 dropout=0.1；(3) 学习率 1e-5→5e-6；(4) 各向同性梯度累积；(5) 数据从40→473条
**效果：** V3 val loss 降至 1.397，泛化良好

### 问题6：Cohere Embed API 403 → 本地 RAG 方案（持续有效）

**现象：** Cohere API Key 没有 Embed 接口权限
**根因：** 免费层级的 API Key 不包含 Embed 能力
**解决：** 切换到 TF-IDF + 余弦相似度的纯本地方案。规范语料从 10 段扩展到 14 段后，仍保持关键词匹配的高精度——GRADE 术语高度明确（"偏倚风险""不精确性""发表偏倚"等），TF-IDF 对专业术语的匹配精度足够。自定义中英文字符 bigram tokenizer 兼顾了"发表偏倚"这类中文复合词的完整匹配和"偏倚"→"偏倚风险"的部分召回。
**效果：** 5 种典型检索场景 Top-1 命中率 100%，零延迟、零成本、完全消除外部 API 依赖。14 段语料仍在本地检索的最佳范围内。

### 问题7：用户等待100秒无反馈

**现象：** 管线用 `graph_app.invoke()` 同步执行，用户看到转圈100+秒
**解决：** `invoke()` → `stream()`，主线程轮询共享 dict 实时更新进度条 + 已完成的阶段性结果
**效果：** 每个阶段（搜索/筛选/提取/合成）完成后立即在 UI 展示中间结果

---

## 五、模拟面试问答

### Q1: 为什么用 LangGraph 而不是简单的顺序调用？

**A:** LangGraph 提供了三个核心价值：

1. **条件路由** — 搜索结果为空时自动跳过筛选，直接到合成节点生成空报告。如果用顺序 if/else，这些条件分支会散落在代码各处，难以追踪。
2. **节点级隔离** — 每个 Agent 被 `_safe_node` 包装，单个节点异常不会导致整个管线崩溃。比如 Cohere API 挂了，Search 节点失败但其余节点仍能执行。
3. **可观测性** — LangGraph 自动向 LangSmith 上报每个节点的输入/输出和延迟，不需要手动埋点。

当然，如果管线是纯粹的顺序执行且没有分支，LangGraph 确实有点重。但一旦有分支 + 需要追踪，它的优势就显现了。

---

### Q2: 你的 Small-to-Large 架构具体是怎么设计的？

**A:** 核心思想是"高频结构化任务用本地小模型，低频复杂推理用云端大模型"。

- **本地侧**：Qwen2.5-1.5B + LoRA 微调处理 PICOS 提取。这是管线中调用频率最高的任务（每篇纳入文献都要做），但任务本身结构化——从摘要中抽取5个字段。1.5B 参数足够，LoRA 只训练了 0.3% 参数就能达到可用精度。好处是零延迟、零 API 成本、数据不出本地。
- **云端侧**：DeepSeek API 处理文献筛选、证据合成、查询扩展。这些任务需要理解复杂医学术语、做多维度判断，1.5B 模型的推理能力不够，必须用大模型。
- **容错**：Extract Agent 实现了三层回退：MLX LoRA → DeepSeek API → 启发式规则。任一层失败自动降级，保证管线不会中断。

---

### Q3: 你们的 RAG 是怎么做的？为什么不用向量数据库？

**A:** 我们在 Synthesis Agent 中做了 RAG——在生成 GRADE 证据报告前，先从 14 段 GRADE/Cochrane 规范知识库中检索最相关的评级标准，注入 prompt。

这个 RAG 模块经历了一次关键的 V1→V2 升级。V1 用了 10 段从搜索引擎拼凑的规范条文，但后来发现三个系统性错误：(1) 把 GRADE 的降级逻辑描述为"5 个域各降 1 级 = 降 5 级"的机械叠加，而 GRADE Book 2024 明确说各域判断与最终等级之间不存在一一对应关系；(2) 没有要求按具体结局分别评级；(3) 缺少对 AI 常见错误（如升降互抵、过度自信结论用语）的防御机制。这些问题直接导致生成的每一份 GRADE 报告在方法论上都有偏差。

V2 完全依据 GRADE Book 2024 + JCE 系列 + ACIP 2024，切成 14 个语义自包含的 chunk。每个 chunk 设计为可独立检索——标题和首句包含核心关键词（"降级域 1：偏倚风险（Risk of Bias）"），正文包含评估清单 + 降级标准 + 遗传关联特殊考量。这样 TF-IDF 无论匹配到标题还是正文都能定位。新增了 3 个关键 chunk：综合判断原则（含正误示例对比）、标准化结论用语模板、AI 必避 8 大错误自查清单。

在 prompt 层面做了三层防御：前置核心原则提醒 + RAG 规范注入 + 输出前 10 项自查清单。这样 LLM 不再是"凭记忆评级"，而是参考权威规范、基于 Python 预计算的观测数据、遵循综合判断原则、最后逐项自查后再输出。

不用向量数据库的原因很务实——14 段语料、几千字中文，GRADE 术语高度明确。自定义的中英文字符 bigram tokenizer 对"发表偏倚"→"发表偏倚降级标准"这种匹配精度很高。sklearn TF-IDF + 余弦相似度，5 种典型检索场景 Top-1 命中率 100%，零延迟零成本零运维。但如果语料扩展到成百上千篇指南全文，就需要上 Embedding + 向量库了。

---

### Q4: 你们的 LoRA 模型是怎么训练的，遇到过什么问题？

**A:** 基座是 Qwen2.5-1.5B-Instruct，用 MLX 框架 + LoRA 在 Apple Silicon M4 上微调。训练数据是通过 DeepSeek API 自动标注的 473 条英文医学摘要，每条标注了 Population、Intervention、Comparison、Outcome、Study_Type 五个字段。

主要踩过的坑：

1. **V1 严重过拟合**：只有40条数据，LoRA rank 设了32太高，没加 dropout。模型在训练集上完美，验证集上输出乱码。解决是 rank 降到8、加 dropout=0.1、学习率从 1e-5 降到 5e-6。

2. **MLX 框架的 `alpha` 参数被弃用**：文档落后于代码，`lora_parameters.yaml` 中必须用 `scale` 代替 `alpha`，否则加载失败。

3. **输出格式不一致**：训练数据中有些 JSON 有空格缩进、有些不带。解决是训练数据统一为紧凑 JSON 格式。

---

### Q5: 你在 Prompt Engineering 上做了哪些关键优化？

**A:** 四个有代表性的优化：

1. **角色效应调整决策边界**：Screen Agent 从"严格筛选专家"改为"宽松筛选专家"，同一 LLM、同一摘要，纳入率从 18% 提升到 70%。这在医学场景特别重要，因为宁可多纳入（让后续环节再判断）也不能漏掉相关文献。

2. **显式语义映射**：在 Screen prompt 中显式写了"precocious puberty、premature thelarche、age at menarche 都视为 puberty timing"，解决了 LLM 按字面匹配而非语义匹配的问题。

3. **反思循环**：被第一轮排除的文献进第二轮——告诉 LLM "你刚才因为XX原因排除了这篇，现在请重新考虑，尤其注意摘要可能省略细节"。实测有 1-2 篇会被成功捞回。

4. **硬约束 vs 软指导，以及"非机械叠加"原则**：RAG 注入的规范对 LLM 最初只是软参考，可以被忽略。但更根本的问题在于——V1 的 GRADE 条文本身就暗示了"5 域各降 1 级 = 降 5 级"的机械叠加逻辑。V2 通过三层防御机制解决了这个问题：

   - **规范层**：用 GRADE Book 2024 官方指南替换，明确"综合判断非机械叠加"原则，并在 `grade_final_rating` chunk 中给出正误示例对比
   - **Prompt 层**：8 条核心原则前置提醒 + RAG 规范注入 + 输出前 10 项自查清单
   - **架构层**：Python 预计算所有观测数据，LLM 不碰数字，只做综合判断和表述

   特别是"证据不足 ≠ 无关联"这条硬约束——极低质量证据只能表述为"任何效应估计都非常不确定"，禁止写"无关联/无影响"。这在医学场景中至关重要，因为"证据不足证明有关联"和"证据证明无关联"是完全不同的结论。

---

### Q6: 你的系统如何处理错误和异常？

**A:** 多层容错机制：

- **API 层**：DeepSeek API 不可用时，`call_chat_model()` 静默返回空字符串，不抛异常
- **节点层**：每个 LangGraph 节点被 `_safe_node()` 包裹，单个节点失败返回 `{"error": "..."}` 而不中断管线
- **任务层**：Extract Agent 的三层回退（MLX → API → Heuristic），Cohere Reranker 的降级（异常时跳过重排序）
- **报告层**：LLM 不可用时使用 `FALLBACK_SYNTHESIS_TEMPLATE` 生成基本报告框架，保留结构化数据供人工查阅
- **API 健康检查**：DeepSeek 可用性检查结果被缓存，避免每次调用都做网络探测

---

### Q7: 为什么选择 MLX 而不是其他推理框架？

**A:** 因为我们的目标设备是 Apple Silicon Mac。MLX 是 Apple 官方的机器学习框架，和 Core ML、ANE 深度集成，在 M4 上推理速度比 llama.cpp 快 30-50%。而且 MLX 的 LoRA 支持很成熟，支持 fused model 导出和 `mlx_lm.generate()` 的流式输出。

如果未来要部署到 NVIDIA GPU 服务器，会考虑 vLLM 或 TensorRT-LLM。但这是工程上的迁移，模型权重是通用的。

---

### Q8: 你的系统有什么局限性？

**A:** 坦诚地说：

1. **检索不完整**：仅检索 PubMed 单一数据库，未查 EMBASE、Cochrane Library、灰色文献。不满足正式系统综述的检索要求。
2. **GRADE 评级受限于摘要**：很多偏倚风险评估需要的信息（随机方法、盲法、ITT分析）在摘要中不会报告，评级可能不准确。
3. **语言限制**：PubMed 检索主要覆盖英文文献，中文文献（如 CNKI、万方）未覆盖。
4. **LoRA 模型精度**：Qwen2.5-1.5B 的 PICOS 提取精度不如 GPT-4o，但通过三层回退机制（本地 → API → 启发式）缓解了这一问题。
5. **Meta 分析/原始研究重叠**：系统目前检测到重叠风险并警告用户，但不能自动排除重复数据。

这些都是我们在文档中明确告知用户的。

---

### Q9: 如果要部署到生产环境，你会做哪些改进？

**A:** 四个方面：

1. **检索完整性**：接入 EMBASE、Cochrane Library API，支持多数据库并行检索和去重
2. **自动 Meta 分析**：当纳入研究≥3篇时，自动计算合并 OR/RR、I² 异质性、生成森林图和漏斗图
3. **人工审核工作流**：在 Extract 和 Synthesis 之间插入人工审核节点，标记低置信度提取结果供复核
4. **API 网关**：引入 API 密钥轮换、速率限制、请求队列，避免生产环境的 API 配额问题

---

### Q10: 在整个项目中，你学到的最重要的经验是什么？

**A:** 三个最深的体会：

1. **不要让 LLM 做它能"看起来能做"但实际做不好的事**——计数、分类、一致性校验。Python 一行 `Counter` 的事，丢给 LLM 只会增加幻觉。三层架构的核心理念就是：Python 负责事实，LLM 负责表述。

2. **Prompt 的角色设定比参数调整更有效**——我们试了 temperature、top_p 的各种组合来调整筛选的严格程度，效果都不明显。最终把角色从"严格排除"改为"宽松筛选"，同一模型行为立刻改变。Prompt engineering 的精髓不是调参数，而是理解 LLM 的角色对齐机制。

3. **工程可观测性是 Agent 系统的必需品而非奢侈品**——在接入 LangSmith 之前，每次 LLM 调用都是黑盒，排查问题只能靠 print。接入之后能看到每一步的 prompt 输入和输出、延迟分布、token 消耗，debug 效率提升了一个数量级。
