from typing import Dict, List
from .bm25 import BM25Retriever
from .vector_store import DenseVectorStore

class HybridRetriever:
    """Hybrid Search combining Sparse BM25 and Dense Vector Embeddings via Reciprocal Rank Fusion."""
    def __init__(self, bm25: BM25Retriever, vector_store: DenseVectorStore, rrf_k: int = 60):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 10) -> List[Dict[str, any]]:
        bm25_results = self.bm25.retrieve(query, top_k=candidate_pool)
        dense_results = self.vector_store.retrieve(query, top_k=candidate_pool)

        rrf_scores: Dict[str, float] = {}
        doc_contents: Dict[str, str] = {}
        bm25_ranks: Dict[str, int] = {}
        dense_ranks: Dict[str, int] = {}

        # 1. Process BM25 ranks
        for item in bm25_results:
            d_id = item["doc_id"]
            rank = item["rank"]
            doc_contents[d_id] = item["content"]
            bm25_ranks[d_id] = rank
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Process Dense ranks
        for item in dense_results:
            d_id = item["doc_id"]
            rank = item["rank"]
            doc_contents[d_id] = item["content"]
            dense_ranks[d_id] = rank
            rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (d_id, score) in enumerate(sorted_docs[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": d_id,
                "score": round(score, 6),
                "bm25_rank": bm25_ranks.get(d_id, None),
                "dense_rank": dense_ranks.get(d_id, None),
                "content": doc_contents[d_id],
                "retriever": "Hybrid_RRF"
            })
        return results
