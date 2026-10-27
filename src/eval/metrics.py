import math
import re
from typing import List, Set

def _normalize_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))

def calculate_context_recall(retrieved_doc_ids: List[str], ground_truth_doc_id: str) -> float:
    """Evaluates if the ground-truth document was successfully recalled in top-k."""
    return 1.0 if ground_truth_doc_id in retrieved_doc_ids else 0.0

def calculate_hit_rate(retrieved_doc_ids: List[str], ground_truth_doc_id: str, k: int = 3) -> float:
    """Calculates Hit Rate@k (1.0 if relevant document is present within top-k candidates, else 0.0)."""
    return 1.0 if ground_truth_doc_id in retrieved_doc_ids[:k] else 0.0

def calculate_context_precision(retrieved_doc_ids: List[str], ground_truth_doc_id: str) -> float:
    """Evaluates how high the relevant document was ranked."""
    if ground_truth_doc_id not in retrieved_doc_ids:
        return 0.0
    rank = retrieved_doc_ids.index(ground_truth_doc_id) + 1
    return round(1.0 / rank, 4)

def calculate_mrr(retrieved_doc_ids: List[str], ground_truth_doc_id: str) -> float:
    """Mean Reciprocal Rank (MRR) for the ground-truth document."""
    return calculate_context_precision(retrieved_doc_ids, ground_truth_doc_id)

def calculate_ndcg(retrieved_doc_ids: List[str], ground_truth_doc_id: str, k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG@k) for binary relevance.

    DCG = 1 / log2(rank + 1) if ground truth is in top-k at 1-based rank, else 0.
    IDCG = 1 / log2(1 + 1) = 1.0 (since only one ideal relevant document exists).
    NDCG = DCG / IDCG.
    """
    if ground_truth_doc_id not in retrieved_doc_ids[:k]:
        return 0.0
    rank = retrieved_doc_ids[:k].index(ground_truth_doc_id) + 1
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0
    return round(dcg / idcg, 4)

def calculate_faithfulness(generated_answer: str, retrieved_contexts: List[str]) -> float:
    """Measures the proportion of statements in the answer grounded in retrieved context (hallucination detector)."""
    ans_tokens = _normalize_tokens(generated_answer)
    if not ans_tokens:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    grounded_tokens = [t for t in ans_tokens if t in combined_context]

    faithfulness_ratio = len(grounded_tokens) / max(1, len(ans_tokens))
    return round(faithfulness_ratio, 4)

def calculate_answer_relevancy(generated_answer: str, expected_answer: str) -> float:
    """Measures semantic token overlap between generated answer and ground-truth answer."""
    gen_tokens = _normalize_tokens(generated_answer)
    exp_tokens = _normalize_tokens(expected_answer)
    if not exp_tokens or not gen_tokens:
        return 0.0

    overlap = gen_tokens.intersection(exp_tokens)
    relevancy = len(overlap) / len(exp_tokens)
    return round(min(1.0, relevancy), 4)
