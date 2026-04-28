"""
数据集扩容脚本
1. 30个多样化检索词覆盖广泛临床领域
2. PubMed检索收集500-600篇原始摘要
3. DeepSeek LLM批量标注PICOS
4. 质量过滤 (完整JSON + 5个字段均非空)
5. 去重+合并现有数据 → 85/15分割 → MLX格式JSONL

预计耗时: 15-20分钟
API费用: <$0.1 (DeepSeek)
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.pubmed_tool import search_and_fetch
from tools.llm import DeepSeekLLM

# ── 30个多样化检索词 ──
SEARCH_QUERIES = [
    # 心血管 (5)
    "ACEI hypertension randomized controlled trial",
    "statins LDL cholesterol cardiovascular RCT",
    "beta blockers heart failure mortality trial",
    "aspirin primary prevention cardiovascular events",
    "calcium channel blocker hypertension elderly RCT",
    # 糖尿病与内分泌 (5)
    "metformin type 2 diabetes clinical trial",
    "SGLT2 inhibitor heart failure diabetes RCT",
    "GLP-1 receptor agonist weight loss obesity trial",
    "insulin glargine type 1 diabetes RCT",
    "DPP-4 inhibitor type 2 diabetes safety RCT",
    # 精神科/神经 (4)
    "SSRI depression randomized controlled trial",
    "atypical antipsychotic schizophrenia RCT",
    "memantine Alzheimer disease randomized trial",
    "levodopa Parkinson disease randomized controlled trial",
    # 消化/呼吸 (4)
    "proton pump inhibitor GERD randomized trial",
    "biologic therapy inflammatory bowel disease RCT",
    "inhaled corticosteroid asthma randomized trial",
    "amoxicillin community acquired pneumonia RCT",
    # 骨科/风湿 (3)
    "bisphosphonate osteoporosis fracture prevention RCT",
    "NSAID osteoarthritis pain randomized trial",
    "biologic DMARD rheumatoid arthritis randomized controlled trial",
    # 肿瘤 (3)
    "immune checkpoint inhibitor non-small cell lung cancer RCT",
    "endocrine therapy breast cancer randomized trial",
    "targeted therapy colorectal cancer randomized controlled trial",
    # 感染/抗生素 (3)
    "antiviral therapy hepatitis C randomized trial",
    "antifungal therapy candidemia RCT",
    "antibiotic surgical site infection prophylaxis randomized trial",
    # 其他重要领域 (3)
    "anticoagulant atrial fibrillation stroke prevention RCT",
    "statin renal outcomes chronic kidney disease trial",
    "tranexamic acid trauma hemorrhage randomized trial",
]

TOTAL_TARGET = 300  # 目标高质量样本数（不是500，按讨论方案）
SYSTEM_PROMPT = "你是医疗实体提取专家，提取PICOS并输出JSON。"

PICOS_ANNOTATION_PROMPT = """你是医疗实体提取专家。从以下摘要中提取PICOS要素。

要求：
- Population: 目标人群（患者类型、样本量、纳入标准）
- Intervention: 干预措施（药物名、剂量、用法）
- Comparison: 对照措施（安慰剂、其他药物、标准治疗）
- Outcome: 结局指标（主要终点、次要终点、测量方式）
- Study_Type: 研究类型（RCT、队列研究、系统评价等）

摘要：
{abstract}

只输出JSON，格式：
{{"Population": "...", "Intervention": "...", "Comparison": "...", "Outcome": "...", "Study_Type": "..."}}"""


def collect_abstracts(max_per_query: int = 20) -> list:
    """检索并收集摘要（30个检索词 × 20篇 = 最多600篇）"""
    all_articles = []
    seen_pmids = set()

    for i, query in enumerate(SEARCH_QUERIES):
        print(f"\n[{i+1}/{len(SEARCH_QUERIES)}] 检索: {query}")
        try:
            articles = search_and_fetch(query, max_results=max_per_query)
        except Exception as e:
            print(f"  检索失败: {e}")
            continue

        new_count = 0
        for a in articles:
            pmid = a.get("pmid", "")
            abstract = a.get("abstract", "")
            if pmid and pmid not in seen_pmids and len(abstract) > 100:
                seen_pmids.add(pmid)
                all_articles.append(a)
                new_count += 1

        print(f"  新增: {new_count}, 累计: {len(all_articles)}")

        if len(all_articles) >= 700:  # 多收集一些以备过滤
            break

        time.sleep(0.5)  # 避免PubMed限速

    print(f"\n收集完成: {len(all_articles)} 篇原始摘要")
    return all_articles


def validate_picos(picos: dict) -> bool:
    """验证PICOS质量"""
    if not isinstance(picos, dict):
        return False
    required = ["Population", "Intervention", "Comparison", "Outcome", "Study_Type"]
    for key in required:
        val = picos.get(key, "")
        if not val or not isinstance(val, str) or len(val.strip()) < 2:
            return False
    # 过滤过短的值（如仅"-"、单个词）
    for key in required:
        if len(picos[key].strip()) < 3:
            return False
    return True


def annotate_all(articles: list, llm) -> list:
    """批量标注PICOS"""
    labeled = []
    failed = 0

    for i, article in enumerate(articles):
        abstract = article.get("abstract", "")
        prompt = PICOS_ANNOTATION_PROMPT.format(abstract=abstract)

        try:
            response = llm.invoke(prompt, temperature=0.1)
            # 提取JSON
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                picos = json.loads(match.group(0))
                if validate_picos(picos):
                    labeled.append({
                        "pmid": article.get("pmid", ""),
                        "title": article.get("title", ""),
                        "abstract": abstract,
                        "picos": picos,
                    })
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  标注进度: {i+1}/{len(articles)}, 成功: {len(labeled)}, 失败: {failed}")

        time.sleep(0.05)  # 轻微间隔，避免API限速

    print(f"\n标注完成: {len(labeled)} 成功, {failed} 失败 ({failed/len(articles)*100:.1f}%)")
    return labeled


def load_existing_labeled(path: str) -> dict:
    """加载已有标注数据，返回 pmid→picos 映射"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["pmid"]: item for item in data}


def to_mlx_format(labeled: list) -> list:
    """转换为MLX训练格式"""
    samples = []
    for item in labeled:
        picos = item["picos"]
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"提取摘要的PICOS：{item['abstract']}"},
                {"role": "assistant", "content": json.dumps(picos, ensure_ascii=False)},
            ]
        })
    return samples


def main():
    print("=" * 60)
    print("Step 1/5: 检索PubMed收集摘要 (30个检索词)")
    print("=" * 60)
    articles = collect_abstracts(max_per_query=20)

    # 保存原始数据
    os.makedirs("data", exist_ok=True)
    with open("data/raw_abstracts_v2.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"原始摘要已保存: data/raw_abstracts_v2.json ({len(articles)} 篇)")

    print(f"\n{'='*60}")
    print("Step 2/5: DeepSeek LLM批量标注PICOS")
    print(f"{'='*60}")
    llm = DeepSeekLLM(model="deepseek-chat", temperature=0.1, max_tokens=512)
    new_labeled = annotate_all(articles, llm)

    print(f"\n{'='*60}")
    print("Step 3/5: 加载已有数据 + 去重合并")
    print(f"{'='*60}")
    existing = load_existing_labeled("data/labeled_picos.json")
    print(f"已有标注: {len(existing)} 条")

    # 合并：新数据覆盖旧数据（如果PMID重复）
    merged_map = dict(existing)
    new_overwrite = 0
    for item in new_labeled:
        pmid = item["pmid"]
        if pmid in merged_map:
            new_overwrite += 1
        merged_map[pmid] = item

    merged = list(merged_map.values())
    print(f"合并后: {len(merged)} 条 (新增: {len(new_labeled) - new_overwrite}, 覆盖: {new_overwrite})")

    # 保存合并后的标注数据
    with open("data/labeled_picos.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"标注数据已保存: data/labeled_picos.json")

    print(f"\n{'='*60}")
    print("Step 4/5: 质量过滤 + 格式转换")
    print(f"{'='*60}")
    mlx_samples = to_mlx_format(merged)
    print(f"MLX格式样本: {len(mlx_samples)} 条")

    # 去重（按abstract内容的hash）
    seen_hashes = set()
    deduped = []
    for s in mlx_samples:
        h = hash(s["messages"][1]["content"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(s)
    print(f"去重后: {len(deduped)} 条 (移除 {len(mlx_samples) - len(deduped)} 条)")

    print(f"\n{'='*60}")
    print("Step 5/5: 85/15分割 → train.jsonl / valid.jsonl")
    print(f"{'='*60}")
    # 固定随机种子保证可复现
    import random
    random.seed(42)
    random.shuffle(deduped)

    split = int(len(deduped) * 0.85)
    train = deduped[:split]
    valid = deduped[split:]

    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open("data/valid.jsonl", "w", encoding="utf-8") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"训练集: {len(train)} 条 → data/train.jsonl")
    print(f"验证集: {len(valid)} 条 → data/valid.jsonl")
    print(f"\n{'='*60}")
    print(f"完成! 总样本: {len(deduped)} (train={len(train)}, valid={len(valid)})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
