import unittest
from src.eval.metrics import (
    calculate_context_precision,
    calculate_context_recall,
    calculate_mrr,
    calculate_hit_rate,
    calculate_ndcg,
    calculate_faithfulness,
    calculate_answer_relevancy,
)
from src.eval.benchmarker import benchmarker

class TestEvalMetrics(unittest.TestCase):
    def test_metrics_calculations(self):
        retrieved = ["doc_a.md", "doc_b.md", "doc_c.md"]
        ground_truth = "doc_b.md"

        # Context Recall
        self.assertEqual(calculate_context_recall(retrieved, ground_truth), 1.0)
        self.assertEqual(calculate_context_recall(retrieved, "missing.md"), 0.0)

        # Hit Rate@k
        self.assertEqual(calculate_hit_rate(retrieved, ground_truth, k=2), 1.0)
        self.assertEqual(calculate_hit_rate(retrieved, ground_truth, k=1), 0.0)
        self.assertEqual(calculate_hit_rate(retrieved, "missing.md", k=3), 0.0)

        # Context Precision / MRR (rank 2 = 1/2 = 0.5)
        self.assertEqual(calculate_context_precision(retrieved, ground_truth), 0.5)
        self.assertEqual(calculate_mrr(retrieved, ground_truth), 0.5)

        # NDCG@k (rank 2 = 1 / log2(3) ≈ 0.6309)
        self.assertEqual(calculate_ndcg(retrieved, ground_truth, k=3), 0.6309)
        self.assertEqual(calculate_ndcg(retrieved, "doc_a.md", k=3), 1.0)
        self.assertEqual(calculate_ndcg(retrieved, "missing.md", k=3), 0.0)

        # Faithfulness
        context = ["PagedAttention uses virtual blocks to prevent memory fragmentation."]
        answer = "PagedAttention prevents memory fragmentation with virtual blocks."
        faith = calculate_faithfulness(answer, context)
        self.assertGreater(faith, 0.8)

    def test_benchmarker_run(self):
        summary = benchmarker.run_benchmark(top_k=3)
        self.assertGreaterEqual(summary["total_test_queries"], 3)
        self.assertEqual(len(summary["summary_table"]), 4)
        self.assertEqual(summary["best_strategy"], "Hybrid_Reranked")

if __name__ == "__main__":
    unittest.main()
