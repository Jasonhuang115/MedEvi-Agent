# MedEvi-Agent

为遗传关联研究选题提供快速的文献筛选与证据评级。

输入**基因/SNP + 疾病名称**，自动完成 PubMed 检索 → PICOS 文献筛选 → 结构化数据提取 → GRADE 证据质量评级报告。

## 效果演示

运行完成后提供三个视图：

| 文献概览 | 数据提取 | 证据评级 |
|---------|---------|---------|
| 纳入/排除决策 + LLM 筛选理由 | PICOS 结构化提取 + 效应量/95%CI | GRADE 逐域评估 + 综合判定报告 |

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/MedEvi-Agent.git
cd MedEvi-Agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx（必填）
# COHERE_API_KEY 和 NCBI_API_KEY 为可选

# 4. 启动
streamlit run app.py
```

> **Mac (Apple Silicon) 用户**：可选安装 `pip install mlx-lm` 启用本地 LoRA 模型加速 PICOS 提取。未安装时自动降级为 API 提取，功能不受影响。

## 架构

```
用户输入 (SNP + 疾病)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Search      │────▶│  Screen      │────▶│  Extract     │────▶│  Synthesis   │
│  Agent       │     │  Agent       │     │  Agent       │     │  Agent       │
│              │     │              │     │              │     │              │
│ · LLM扩展检索词 │     │ · PICOS筛选   │     │ · MLX LoRA提取│     │ · Python预统计 │
│ · PubMed检索  │     │ · 反思轮捞回   │     │ · DeepSeek回退│     │ · RAG检索规范 │
│ · Cohere重排  │     │ · Fallback规则│     │ · 数值数据提取 │     │ · LLM综合判断 │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  LangGraph StateGraph · Pydantic 类型安全状态 · LangSmith 全链路追踪
```

### 三层证据合成架构（Synthesis Agent）

| 层级 | 职责 | 实现 |
|------|------|------|
| Layer 1 | 统计量预计算 | Python 完成全部计数、分类、GRADE 预评估 |
| Layer 2 | 规范约束注入 | RAG 检索 GRADE Book 2024 条文作为硬约束 |
| Layer 3 | 报告撰写 | LLM 解读预计算数据，在规范约束下综合判断 |

### 混合推理（Extract Agent）

```
MLX LoRA (本地Apple Silicon) → DeepSeek API → 启发式规则
      主推理 (~1.85s/篇)          回退              保底
```

## 技术栈

`Python` · `LangGraph` · `LangChain` · `Pydantic v2` · `LangSmith`
`DeepSeek-V3 API` · `MLX` · `LoRA (Qwen2.5-1.5B)` · `Cohere Rerank v3`
`PubMed E-utilities` · `TF-IDF + 余弦相似度 RAG` · `Streamlit`
`ChromaDB` · `sentence-transformers` · `scikit-learn`

### LoRA 微调

- 基座模型：Qwen2.5-1.5B-Instruct
- 配置：rank=8, alpha=16, ~5.28M 可训参数 (0.34%)
- 训练：Apple Silicon M4, 3 轮迭代 (V1→V2→V3)
- 推理框架：MLX (`mlx_lm`)

### RAG 知识库

- 14 条 GRADE 规范语义块（GRADE Book 2024 + JCE 2011 + ACIP 2024）
- 自研中英混合分词器（CJK 二元组 + 英文单词保留）
- TF-IDF 向量化 + 余弦相似度检索

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API Key（免费注册即送额度） |
| `COHERE_API_KEY` | 可选 | Cohere Rerank API Key |
| `NCBI_API_KEY` | 可选 | PubMed E-utilities API Key（提升检索速率） |
| `LANGCHAIN_API_KEY` | 可选 | LangSmith 追踪 |
| `MLX_PICOS_MODEL_PATH` | 可选 | 本地 LoRA 模型路径 |

## 免责声明

所有 AI 生成结果仅供科研参考，不替代系统性文献评价和人工复核。建议在正式 Meta 分析中使用 Stata、RevMan 等专业统计软件。

## License

MIT
