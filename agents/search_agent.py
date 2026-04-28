"""Search Agent: PubMed retrieval + Cohere rerank (no HuggingFace dependency)."""
from __future__ import annotations

from typing import Any, Dict, List

from agents.common import build_state_patch, state_get
from agents.llm_router import call_chat_model
from prompts.search_prompt import QUERY_EXPANSION_PROMPT
from tools.pubmed_tool import search_pubmed, fetch_abstracts


def _expand_query(raw_query: str) -> str:
    """Use LLM to rewrite a natural-language query into a PubMed Boolean query."""
    prompt = QUERY_EXPANSION_PROMPT.format(query=raw_query)
    expanded = call_chat_model(prompt, temperature=0.0)
    if expanded and 10 < len(expanded) < 300:
        return expanded.strip()
    return raw_query


def search_agent(state: Any) -> Dict[str, Any]:
    query = state_get(state, "query", "")

    if not query:
        return build_state_patch(error="query 不能为空")

    # 用 LLM 将自然语言改写为 PubMed Boolean 查询（加同义词）
    pubmed_query = _expand_query(query)
    print(f"原始查询: {query}")
    print(f"PubMed查询: {pubmed_query}")

    pmids = search_pubmed(pubmed_query, max_results=50)
    if not pmids:
        return build_state_patch(
            pubmed_ids=[],
            raw_abstracts=[],
            reranked_abstracts=[],
            error="未检索到相关文献",
        )

    abstracts = fetch_abstracts(pmids)

    # 用原始自然语言 query（非 PubMed 查询）做 rerank，语义匹配更准
    try:
        from tools.reranker import rerank_abstracts

        reranked = rerank_abstracts(query, abstracts, top_k=20)
    except Exception:
        reranked = abstracts[:20]

    return build_state_patch(
        pubmed_ids=pmids,
        raw_abstracts=abstracts,
        reranked_abstracts=reranked,
        error="",
    )
