"""
Cohere Reranker - 文献重排序工具
"""
import os
from typing import List, Dict
import cohere
from dotenv import load_dotenv

load_dotenv()


class CohereReranker:
    """Cohere重排序器"""

    def __init__(self, model: str = "rerank-english-v3.0"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 COHERE_API_KEY")
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
        text_key: str = "abstract"
    ) -> List[Dict]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表，每个文档是字典
            top_k: 返回前k个结果
            text_key: 文档中用于排序的文本字段名

        Returns:
            重排序后的文档列表，按相关性降序
        """
        if not documents:
            return []

        # 提取文本
        docs_text = [doc.get(text_key, "") for doc in documents]

        # 调用Cohere Rerank API
        results = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs_text,
            top_n=min(top_k, len(documents))
        )

        # 构建返回结果
        reranked = []
        for item in results.results:
            doc = documents[item.index].copy()
            doc["rerank_score"] = item.relevance_score
            reranked.append(doc)

        return reranked


# 全局实例
_reranker = None


def get_reranker() -> CohereReranker:
    """获取重排序器实例（单例）"""
    global _reranker
    if _reranker is None:
        _reranker = CohereReranker()
    return _reranker


def rerank_abstracts(
    query: str,
    documents: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """
    对摘要进行重排序（便捷函数）

    Args:
        query: 查询文本
        documents: 文档列表
        top_k: 返回前k个结果

    Returns:
        重排序后的文档列表
    """
    reranker = get_reranker()
    return reranker.rerank(query, documents, top_k)
