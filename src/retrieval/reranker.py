import re
from typing import Dict, List

class CrossEncoderReranker:
    """Simulates a Cross-Encoder reranker evaluating query-document cross-attention."""
    def __init__(self):
        pass

    def rerank(self, query: str, candidates: List[Dict[str, any]], top_k: int = 3) -> List[Dict[str, any]]:
        query_words = set(re.findall(r"\w+", query.lower()))
        reranked = []

        for item in candidates:
            content = item["content"].lower()
            # 1. Exact query terms presence ratio
            matched_terms = [w for w in query_words if w in content]
            term_coverage = len(matched_terms) / max(1, len(query_words))

            # 2. Sequential phrase match boost
            clean_query = " ".join(query.lower().split()[:3])
            phrase_bonus = 0.3 if clean_query in content else 0.0

            # 3. First stage baseline score
            initial_score = item.get("score", 0.5)

            # Combined Cross-Encoder score
            cross_score = (term_coverage * 0.6) + phrase_bonus + (initial_score * 0.1)

            updated = dict(item)
            updated["initial_rank"] = item["rank"]
            updated["cross_encoder_score"] = round(cross_score, 4)
            updated["retriever"] = "Hybrid_RRF_Reranked"
            reranked.append(updated)

        # Sort descending by cross encoder score
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        for rank, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = rank

        return reranked[:top_k]
