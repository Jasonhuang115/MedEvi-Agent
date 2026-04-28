"""
从 raw_abstracts_v2.json 续跑：标注PICOS + 质量过滤 + 生成MLX格式数据
"""
import json, os, re, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.llm import DeepSeekLLM

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


def validate_picos(picos: dict) -> bool:
    if not isinstance(picos, dict):
        return False
    for key in ["Population", "Intervention", "Comparison", "Outcome", "Study_Type"]:
        val = picos.get(key, "")
        if not val or not isinstance(val, str) or len(val.strip()) < 3:
            return False
    return True


# ═══ Step 1: 加载已有数据 ═══
print("Step 1: 加载数据")
with open("data/raw_abstracts_v2.json", encoding="utf-8") as f:
    articles = json.load(f)
print(f"  原始摘要: {len(articles)} 篇")

existing_map = {}
if os.path.exists("data/labeled_picos.json"):
    with open("data/labeled_picos.json", encoding="utf-8") as f:
        for item in json.load(f):
            existing_map[item["pmid"]] = item
print(f"  已有标注: {len(existing_map)} 条")

to_annotate = [a for a in articles if a.get("pmid", "") not in existing_map]
print(f"  待标注: {len(to_annotate)} 篇")

# ═══ Step 2: 批量标注 ═══
print("\nStep 2: DeepSeek LLM 批量标注PICOS")
llm = DeepSeekLLM(model="deepseek-chat", temperature=0.1, max_tokens=512)

labeled = list(existing_map.values())
failed = 0
save_interval = 50

for i, article in enumerate(to_annotate):
    abstract = article.get("abstract", "")

    try:
        response = llm.invoke(PICOS_ANNOTATION_PROMPT.format(abstract=abstract), temperature=0.1)
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

    # 定期保存 + 打印进度
    if (i + 1) % save_interval == 0:
        with open("data/labeled_picos.json", "w", encoding="utf-8") as f:
            json.dump(labeled, f, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{len(to_annotate)}] 已保存: {len(labeled)} 条 (失败: {failed})")

    time.sleep(0.05)

# 最终保存
with open("data/labeled_picos.json", "w", encoding="utf-8") as f:
    json.dump(labeled, f, ensure_ascii=False, indent=2)
print(f"  完成! 总计: {len(labeled)} 条, 失败: {failed}")

# ═══ Step 3: 转换为MLX格式 ═══
print("\nStep 3: 转换为MLX格式")
mlx_samples = []
for item in labeled:
    picos = item["picos"]
    mlx_samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"提取摘要的PICOS：{item['abstract']}"},
            {"role": "assistant", "content": json.dumps(picos, ensure_ascii=False)},
        ]
    })

# 去重
seen_hashes = set()
deduped = []
for s in mlx_samples:
    h = hash(s["messages"][1]["content"])
    if h not in seen_hashes:
        seen_hashes.add(h)
        deduped.append(s)
print(f"  去重后: {len(deduped)} 条 (移除: {len(mlx_samples)-len(deduped)})")

# ═══ Step 4: 85/15分割 ═══
print("\nStep 4: 85/15 训练/验证分割")
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

print(f"\n{'='*60}")
print(f"完成! train={len(train)}, valid={len(valid)}, total={len(deduped)}")
print(f"{'='*60}")
