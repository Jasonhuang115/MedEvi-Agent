"""
数据集准备脚本
1. 检索多个病症的文献
2. LLM标注PICOS
3. 格式化JSONL
"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

from tools.pubmed_tool import search_and_fetch
from tools.llm import get_llm

# 多个不同病症的检索词，确保数据多样性
SEARCH_QUERIES = [
    "ACEI hypertension randomized controlled trial",
    "metformin diabetes type 2 clinical trial",
    "statins LDL cholesterol randomized trial",
    "aspirin primary prevention cardiovascular",
    "beta blockers heart failure mortality RCT",
    "PPI gastroesophageal reflux disease RCT",
    "SSRI depression randomized controlled trial",
    "ibuprofen pain management RCT",
    "amoxicillin pneumonia randomized trial",
    "alendronate osteoporosis fracture RCT",
]


def collect_abstracts(max_per_query: int = 15) -> list:
    """检索并收集摘要"""
    all_articles = []
    seen_pmids = set()

    for query in SEARCH_QUERIES:
        print(f"\n检索: {query}")
        articles = search_and_fetch(query, max_results=max_per_query)

        for a in articles:
            pmid = a["pmid"]
            if pmid not in seen_pmids and len(a.get("abstract", "")) > 100:
                seen_pmids.add(pmid)
                all_articles.append(a)

        print(f"  当前总计: {len(all_articles)} 篇")

        if len(all_articles) >= 120:
            break

    print(f"\n收集完成，共 {len(all_articles)} 篇文献")
    return all_articles


def annotate_picos(abstract: str, llm) -> dict:
    """调用LLM标注PICOS"""
    prompt = f"""你是医疗实体提取专家。从以下摘要中提取PICOS要素。

要求：
- Population: 目标人群
- Intervention: 干预措施
- Comparison: 对照措施
- Outcome: 结局指标
- Study_Type: 研究类型

摘要：
{abstract}

只输出JSON，格式：
{{"Population": "...", "Intervention": "...", "Comparison": "...", "Outcome": "...", "Study_Type": "..."}}"""

    response = llm.invoke(prompt, temperature=0.1)

    # 尝试提取JSON
    import re
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None


if __name__ == "__main__":
    # Step 1: 收集文献
    articles = collect_abstracts()

    # 保存原始数据
    os.makedirs("data", exist_ok=True)
    with open("data/raw_abstracts.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"原始摘要已保存: data/raw_abstracts.json")
