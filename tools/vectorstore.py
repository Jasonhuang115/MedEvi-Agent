"""
ChromaDB向量存储工具
用于文献摘要的向量检索
"""
import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class ChromaStore:
    """ChromaDB向量存储管理器（延迟加载embedding模型）"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "pubmed_abstracts"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._initialized = False
        self.embedding_fn = None
        self.client = None
        self.collection = None

    def _ensure_initialized(self):
        """延迟初始化：只在首次添加或搜索文档时才加载embedding模型。"""
        if self._initialized:
            return

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        self._initialized = True
        print(f"ChromaDB初始化完成，集合: {self.collection_name}")

    def add_documents(self, documents: List[Dict]) -> int:
        if not documents:
            return 0

        self._ensure_initialized()

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            pmid = doc.get("pmid", "")
            if not pmid:
                continue

            # 组合标题和摘要作为文本
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            text = f"Title: {title}\n\nAbstract: {abstract}"

            ids.append(pmid)
            texts.append(text)

            # 存储元数据
            metadata = {
                "pmid": pmid,
                "title": title[:500],  # 截断防止过长
                "year": doc.get("year", ""),
                "journal": doc.get("journal", ""),
            }
            metadatas.append(metadata)

        if not ids:
            return 0

        # 检查已存在的文档，避免重复添加
        existing = self.collection.get(ids=ids)
        existing_ids = set(existing["ids"]) if existing["ids"] else set()

        # 只添加新文档
        new_ids = []
        new_texts = []
        new_metadatas = []

        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_ids.append(doc_id)
                new_texts.append(texts[i])
                new_metadatas.append(metadatas[i])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metadatas
            )
            print(f"添加 {len(new_ids)} 篇新文档到向量库")

        return len(new_ids)

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        self._ensure_initialized()

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if not results["ids"] or not results["ids"][0]:
            return documents

        for i, doc_id in enumerate(results["ids"][0]):
            doc = {
                "pmid": doc_id,
                "title": results["metadatas"][0][i].get("title", ""),
                "year": results["metadatas"][0][i].get("year", ""),
                "journal": results["metadatas"][0][i].get("journal", ""),
                "abstract": results["documents"][0][i].replace("Title: ", "").split("\n\nAbstract: ")[-1] if "\n\nAbstract: " in results["documents"][0][i] else results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            }
            documents.append(doc)

        return documents

    def delete_all(self):
        """清空集合"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
            print(f"已删除 {len(all_ids)} 篇文档")

    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()


# 全局实例
_store = None


def get_store(persist_dir: str = "./chroma_db") -> ChromaStore:
    """获取向量存储实例（单例）"""
    global _store
    if _store is None:
        _store = ChromaStore(persist_dir=persist_dir)
    return _store


def add_to_vectorstore(documents: List[Dict]) -> int:
    """便捷函数：添加文档"""
    store = get_store()
    return store.add_documents(documents)


def search_vectorstore(query: str, top_k: int = 20) -> List[Dict]:
    """便捷函数：搜索文档"""
    store = get_store()
    return store.search(query, top_k)
