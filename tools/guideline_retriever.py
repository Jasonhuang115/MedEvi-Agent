"""Guideline retriever: TF-IDF + cosine similarity for GRADE/Cochrane RAG.

Uses sklearn TfidfVectorizer for local, zero-API retrieval. The guideline corpus
is small (10 chunks) and uses well-defined terminology, so keyword overlap works
well without needing an embedding API.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.guideline_store import GUIDELINE_CHUNKS

# A custom tokenizer that keeps Chinese characters as-is (sklearn's default
# tokenizer splits on whitespace and discards single characters, which would
# destroy Chinese text). We use a simple character n-gram approach.
def _char_tokenizer(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text into unigrams and bigrams."""
    # Split by whitespace for English words
    tokens = []
    for word in text.split():
        # For Chinese-heavy segments, also extract character bigrams
        chinese_chars = [c for c in word if '一' <= c <= '鿿']
        if len(chinese_chars) > 2:
            tokens.append(word)  # keep original
            # Add character bigrams
            for i in range(len(chinese_chars) - 1):
                tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        else:
            tokens.append(word)
    return tokens


class GuidelineRetriever:
    """Retrieve relevant GRADE/Cochrane guidelines using TF-IDF."""

    def __init__(self):
        self._chunks = GUIDELINE_CHUNKS
        self._texts = [c["content"] for c in GUIDELINE_CHUNKS]
        self._vectorizer = TfidfVectorizer(
            tokenizer=_char_tokenizer,
            max_features=500,
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant guideline chunks for the given query."""
        if not query:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:  # minimal relevance threshold
                results.append({
                    "title": self._chunks[idx]["title"],
                    "content": self._chunks[idx]["content"],
                    "score": round(float(scores[idx]), 3),
                })
        return results


# Singleton
_retriever: Optional[GuidelineRetriever] = None


def get_retriever() -> GuidelineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GuidelineRetriever()
    return _retriever


def retrieve_guidelines(query: str, top_k: int = 3) -> List[Dict]:
    return get_retriever().retrieve(query, top_k)
