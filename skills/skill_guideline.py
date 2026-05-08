"""Skill: RAG 规范检索。独立于 LangGraph state。"""


def retrieve_guidelines(topic: str, top_k: int = 4) -> list[dict]:
    """从 GRADE/Cochrane 权威规范知识库检索相关条文。

    Returns:
        [{title, content}]
    """
    try:
        from tools.guideline_retriever import retrieve_guidelines as _retrieve

        return _retrieve(topic, top_k=top_k)
    except Exception:
        return []
