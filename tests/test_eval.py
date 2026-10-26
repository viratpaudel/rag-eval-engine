import pytest
from src.eval.metrics import (
    calculate_context_precision,
    calculate_context_recall,
    calculate_mrr,
    calculate_faithfulness,
    calculate_answer_relevancy,
)
from src.eval.benchmarker import benchmarker

def test_metrics_calculations():
    retrieved = ["doc_a.md", "doc_b.md", "doc_c.md"]
    ground_truth = "doc_b.md"

    # Context Recall
    assert calculate_context_recall(retrieved, ground_truth) == 1.0
    assert calculate_context_recall(retrieved, "missing.md") == 0.0

    # Context Precision / MRR (rank 2 = 1/2 = 0.5)
    assert calculate_context_precision(retrieved, ground_truth) == 0.5
    assert calculate_mrr(retrieved, ground_truth) == 0.5

    # Faithfulness
    context = ["PagedAttention uses virtual blocks to prevent memory fragmentation."]
    answer = "PagedAttention prevents memory fragmentation with virtual blocks."
    faith = calculate_faithfulness(answer, context)
    assert faith > 0.8

def test_benchmarker_run():
    summary = benchmarker.run_benchmark(top_k=3)
    assert summary["total_test_queries"] >= 3
    assert len(summary["summary_table"]) == 4
    assert summary["best_strategy"] == "Hybrid_Reranked"
