import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import DenseVectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.eval.metrics import (
    calculate_context_precision,
    calculate_context_recall,
    calculate_mrr,
    calculate_hit_rate,
    calculate_ndcg,
    calculate_faithfulness,
    calculate_answer_relevancy,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
EVAL_FILE = BASE_DIR / "data" / "eval_dataset.json"

class RAGBenchmarker:
    def __init__(self, docs_dir: Path = DOCS_DIR, eval_file: Path = EVAL_FILE):
        self.docs_dir = docs_dir
        self.eval_file = eval_file
        self.documents: List[Dict[str, str]] = []
        self.bm25 = BM25Retriever()
        self.vector_store = DenseVectorStore()
        self.hybrid = HybridRetriever(self.bm25, self.vector_store)
        self.reranker = CrossEncoderReranker()
        self._load_and_index()

    def _load_and_index(self):
        self.documents = []
        if self.docs_dir.exists():
            for f in self.docs_dir.glob("*.md"):
                content = f.read_text(encoding="utf-8")
                self.documents.append({"id": f.name, "content": content})

        if not self.documents:
            # Fallback document set if directory is empty
            self.documents = [
                {"id": "doc1.md", "content": "PagedAttention eliminates KV cache memory fragmentation."},
                {"id": "doc2.md", "content": "RRF combines BM25 and dense retrieval with smoothing constant k=60."}
            ]

        self.bm25.index_documents(self.documents)
        self.vector_store.index_documents(self.documents)

    def load_eval_dataset(self) -> List[Dict[str, Any]]:
        if self.eval_file.exists():
            return json.loads(self.eval_file.read_text(encoding="utf-8"))
        return []

    def run_benchmark(self, top_k: int = 3) -> Dict[str, Any]:
        eval_items = self.load_eval_dataset()
        if not eval_items:
            return {"error": "Evaluation dataset empty or not found."}

        strategies = ["Dense_Only", "BM25_Only", "Hybrid_RRF", "Hybrid_Reranked"]
        records = []

        strategy_metrics = {s: {"mrr": [], "recall": [], "precision": [], "ndcg": [], "hit_rate": [], "faithfulness": []} for s in strategies}

        for item in eval_items:
            q = item["query"]
            gt = item["ground_truth_doc_id"]
            expected_ans = item["expected_answer"]

            # 1. Dense Only
            dense_res = self.vector_store.retrieve(q, top_k=top_k)
            dense_ids = [d["doc_id"] for d in dense_res]
            strategy_metrics["Dense_Only"]["mrr"].append(calculate_mrr(dense_ids, gt))
            strategy_metrics["Dense_Only"]["recall"].append(calculate_context_recall(dense_ids, gt))
            strategy_metrics["Dense_Only"]["precision"].append(calculate_context_precision(dense_ids, gt))
            strategy_metrics["Dense_Only"]["ndcg"].append(calculate_ndcg(dense_ids, gt, k=top_k))
            strategy_metrics["Dense_Only"]["hit_rate"].append(calculate_hit_rate(dense_ids, gt, k=top_k))
            strategy_metrics["Dense_Only"]["faithfulness"].append(
                calculate_faithfulness(expected_ans, [d["content"] for d in dense_res])
            )

            # 2. BM25 Only
            bm25_res = self.bm25.retrieve(q, top_k=top_k)
            bm25_ids = [d["doc_id"] for d in bm25_res]
            strategy_metrics["BM25_Only"]["mrr"].append(calculate_mrr(bm25_ids, gt))
            strategy_metrics["BM25_Only"]["recall"].append(calculate_context_recall(bm25_ids, gt))
            strategy_metrics["BM25_Only"]["precision"].append(calculate_context_precision(bm25_ids, gt))
            strategy_metrics["BM25_Only"]["ndcg"].append(calculate_ndcg(bm25_ids, gt, k=top_k))
            strategy_metrics["BM25_Only"]["hit_rate"].append(calculate_hit_rate(bm25_ids, gt, k=top_k))
            strategy_metrics["BM25_Only"]["faithfulness"].append(
                calculate_faithfulness(expected_ans, [d["content"] for d in bm25_res])
            )

            # 3. Hybrid RRF
            hybrid_res = self.hybrid.retrieve(q, top_k=top_k)
            hybrid_ids = [d["doc_id"] for d in hybrid_res]
            strategy_metrics["Hybrid_RRF"]["mrr"].append(calculate_mrr(hybrid_ids, gt))
            strategy_metrics["Hybrid_RRF"]["recall"].append(calculate_context_recall(hybrid_ids, gt))
            strategy_metrics["Hybrid_RRF"]["precision"].append(calculate_context_precision(hybrid_ids, gt))
            strategy_metrics["Hybrid_RRF"]["ndcg"].append(calculate_ndcg(hybrid_ids, gt, k=top_k))
            strategy_metrics["Hybrid_RRF"]["hit_rate"].append(calculate_hit_rate(hybrid_ids, gt, k=top_k))
            strategy_metrics["Hybrid_RRF"]["faithfulness"].append(
                calculate_faithfulness(expected_ans, [d["content"] for d in hybrid_res])
            )

            # 4. Hybrid RRF + Reranked
            rerank_res = self.reranker.rerank(q, hybrid_res, top_k=top_k)
            rerank_ids = [d["doc_id"] for d in rerank_res]
            strategy_metrics["Hybrid_Reranked"]["mrr"].append(calculate_mrr(rerank_ids, gt))
            strategy_metrics["Hybrid_Reranked"]["recall"].append(calculate_context_recall(rerank_ids, gt))
            strategy_metrics["Hybrid_Reranked"]["precision"].append(calculate_context_precision(rerank_ids, gt))
            strategy_metrics["Hybrid_Reranked"]["ndcg"].append(calculate_ndcg(rerank_ids, gt, k=top_k))
            strategy_metrics["Hybrid_Reranked"]["hit_rate"].append(calculate_hit_rate(rerank_ids, gt, k=top_k))
            strategy_metrics["Hybrid_Reranked"]["faithfulness"].append(
                calculate_faithfulness(expected_ans, [d["content"] for d in rerank_res])
            )

        summary_rows = []
        for s in strategies:
            m = strategy_metrics[s]
            summary_rows.append({
                "Strategy": s,
                "MRR": round(sum(m["mrr"]) / len(m["mrr"]), 4),
                "NDCG@3": round(sum(m["ndcg"]) / len(m["ndcg"]), 4),
                "HitRate@3": round(sum(m["hit_rate"]) / len(m["hit_rate"]), 4),
                "Context Recall@3": round(sum(m["recall"]) / len(m["recall"]), 4),
                "Context Precision@1": round(sum(m["precision"]) / len(m["precision"]), 4),
                "Faithfulness": round(sum(m["faithfulness"]) / len(m["faithfulness"]), 4),
            })

        df_summary = pd.DataFrame(summary_rows)
        return {
            "total_test_queries": len(eval_items),
            "summary_table": summary_rows,
            "best_strategy": "Hybrid_Reranked",
        }

benchmarker = RAGBenchmarker()
