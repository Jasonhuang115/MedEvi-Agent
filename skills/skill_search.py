"""Skill: PubMed 检索 + Cohere 语义重排。独立于 LangGraph state。"""


def search(snp: str, disease: str, max_results: int = 50) -> list[dict]:
    """检索与给定 SNP 和疾病相关的 PubMed 文献。

    Returns:
        [{pmid, title, abstract, relevance_score}]
    """
    from agents.search_agent import _expand_query
    from tools.pubmed_tool import search_pubmed, fetch_abstracts
    from tools.reranker import rerank_abstracts

    query = _expand_query(f"{snp} polymorphism and {disease} risk")
    pmids = search_pubmed(query, max_results=max_results)
    if not pmids:
        return []

    abstracts = fetch_abstracts(pmids)
    try:
        return rerank_abstracts(f"{snp} {disease}", abstracts, top_k=20)
    except Exception:
        return abstracts[:20]
