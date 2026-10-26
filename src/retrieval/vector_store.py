import os
import math
import hashlib
from typing import Dict, List, Tuple

def _dense_hash_embedding(text: str, dim: int = 64) -> List[float]:
    """Generates a normalized dense vector embedding deterministically."""
    words = text.lower().split()
    vec = [0.0] * dim
    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> (i % 32)) & 1
            val = 1.0 if bit == 1 else -1.0
            vec[i] += val

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot))

class DenseVectorStore:
    """In-memory dense vector store with cosine similarity retrieval."""
    def __init__(self, dimension: int = 64):
        self.dim = dimension
        self.doc_ids: List[str] = []
        self.contents: List[str] = []
        self.embeddings: List[List[float]] = []

    def index_documents(self, documents: List[Dict[str, str]]):
        self.doc_ids = []
        self.contents = []
        self.embeddings = []

        for doc in documents:
            self.doc_ids.append(doc["id"])
            self.contents.append(doc["content"])
            emb = _dense_hash_embedding(doc["content"], self.dim)
            self.embeddings.append(emb)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        query_emb = _dense_hash_embedding(query, self.dim)
        scores: List[Tuple[int, float]] = []

        for idx, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_emb, doc_emb)
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, sim) in enumerate(scores[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": self.doc_ids[idx],
                "score": round(sim, 4),
                "content": self.contents[idx],
                "retriever": "Dense_Cosine"
            })
        return results
