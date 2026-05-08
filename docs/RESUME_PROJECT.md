# MedEvi-Agent — 遗传关联研究自动证据合成系统

## 一、项目背景

在循证医学领域，完成一项遗传关联研究的系统综述通常需要数周至数月：检索多个数据库、按PICOS标准筛选文献、提取效应量数据、按GRADE规范评级证据质量。这一流程高度依赖专业人员手工完成，且评级过程容易出现主观偏差。MedEvi-Agent 将这一完整流程自动化：用户仅需输入**基因/SNP + 疾病名称**，系统即可自动完成 PubMed 检索、PICOS 文献筛选、结构化数据提取和 GRADE 证据质量评级报告生成，将数周的综述工作压缩至 2–3 分钟。

## 二、技术栈

**后端框架与编排：** Python · LangGraph（StateGraph 有向图编排 + 条件路由） · LangChain · Pydantic v2（TypedDict → BaseModel 类型安全状态管理） · LangSmith（全链路 LLM 调用追踪）

**AI 模型与推理：**
- **云端：** DeepSeek-V3 API（OpenAI 兼容协议，承担文献筛选、PICOS 扩展、数值提取、报告生成）
- **端侧：** Qwen2.5-1.5B-Instruct + LoRA（rank=8, ~5.28M 可训参数）微调后在 Apple Silicon M4 上通过 MLX 本地推理，作为 PICOS 提取的主力模型（~1.85s/篇，零 API 成本）
- **混合推理架构（Small-to-Large）：** MLX LoRA → DeepSeek API → 启发式规则，三级降级保底

**信息检索与 RAG：**
- PubMed E-utilities API（esearch + efetch，XML 解析，支持 API Key 限速控制）
- Cohere Rerank v3.0 语义重排序（提升检索相关性）
- 自建 GRADE 指南知识库（GRADE Book 2024 + JCE 2011 + ACIP 2024，14 条语义完备块）
- 自研中英混合分词器（CJK 字符二元组 + 英文单词保留）配合 `TfidfVectorizer` 实现余弦相似度检索（Top-1 命中率 100%）

**前端：** Streamlit（实时流式进度、卡片式文献展示、多 Tab 结果浏览、CSV/Markdown 导出）

**其他：** ChromaDB（向量存储，sentence-transformers 嵌入） · Cohere Python SDK · scikit-learn · NumPy · threading（异步流水线 + 轮询 UI）

## 三、项目内容（STAR）

**Situation：** 遗传关联研究的系统综述依赖人工完成 PubMed 检索、PICOS 筛选、效应量提取和 GRADE 证据评级，流程繁琐、耗时长，且 GRADE 降级规则常被机械叠加而非综合判断，导致评级偏差。

**Task：** 构建一个全自动证据合成 Agent 系统，实现从 SNP/疾病输入到 GRADE 评级报告的端到端自动化，同时确保：(1) 筛选敏感度优于召回率（宁宽勿严）；(2) 数据统计在 Python 层完成（非 LLM 计数）；(3) GRADE 评级遵循"综合判断非机械叠加"核心原则。

**Action：**

1. **多 Agent 流水线设计** — 基于 LangGraph 构建 4 节点 StateGraph：Search（LLM 扩展检索词 → PubMed esearch/efetch → Cohere Rerank 语义重排序）→ Screen（LLM 逐篇 PICOS 筛选 + 反思轮捞回被排除文献 + API 异常时关键词 Fallback）→ Extract（MLX LoRA 本地提取 PICOS → DeepSeek 独立提取数值结局 → 置信度标记 + 复核标记）→ Synthesis（三层架构：Python 预计算全部统计量 → RAG 检索 GRADE 规范条文作为硬约束 → LLM 综合判断生成报告）。每个节点由 `_safe_node` 包裹实现故障隔离，条件路由在无结果/无通过文献时跳过后续节点。

2. **LoRA 微调与混合推理** — 在 Apple Silicon M4 上用 MLX 对 Qwen2.5-1.5B 进行 LoRA 微调（r=8, α=16），经 3 轮迭代（V1: 40 样本严重过拟合 val_loss=3.53 → V2: 调整 rank+dropout+lr 正常收敛 val_loss=1.16 → V3: 473 样本生产级 val_loss=1.40），最终部署为本地 PICOS 提取模型。Extract Agent 采用 Small-to-Large 三级降级策略：优先 MLX LoRA 本地推理（零成本）→ API 不可用时切 DeepSeek → 完全离线时用启发式规则。

3. **RAG 知识库与三层防御** — 将 GRADE Book 2024 全文拆分为 14 条语义自包含块（含偏倚风险/不一致性/间接性/不精确性/发表偏倚 5 降级域 + 3 升级因素 + 最终判定逻辑等），自研中英混合分词器实现 TF-IDF 余弦相似度检索。三层防御体系：(1) Python 层预计算所有统计量，杜绝 LLM 计数错误；(2) RAG 检索 GRADE 规范条文作为报告生成的硬约束；(3) Prompt 内嵌 8 条核心原则 + 10 项输出前自查清单，并强制禁止开场白/第一人称（"好的""遵照指示"等）。

4. **工程化与迭代** — 通过 LangSmith 追踪全链路 LLM 调用延迟与 Token 消耗；Streamlit 前端支持实时流式进度轮询、卡片式文献展示（含置信度颜色标记 + ⚠️ 复核标记）、中文标签化 PICOS 表格、三 Tab 结果浏览（文献概览/数据提取/证据评级）及 CSV/Markdown 导出；基于用户反馈完成 6 轮 UI 迭代（简化输入、去除内部架构暴露、修复数据源 Bug、优化表格展示等）；编写 10 条跨疾病领域查询的批量自动化测试脚本。

**Result：**
- 端到端自动化流程：输入 SNP + 疾病 → 2–3 分钟输出 GRADE 证据评级报告
- 文献筛选：反思轮平均捞回 0–2 篇被误排文献，API 异常时 Fallback 规则保障可用性
- PICOS 提取：MLX LoRA 本地推理覆盖 >90% 调用（~1.85s/篇），单次运行节省约 $0.15 API 费用
- RAG 检索：5 类查询场景 Top-1 命中率 100%，GRADE 评级从"机械叠加降级次数"修正为"综合判断"
- LoRA 模型：3 轮迭代，V3 生产模型验证损失 1.40，可训参数仅占基座模型 0.34%
