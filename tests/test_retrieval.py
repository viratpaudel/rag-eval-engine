import unittest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import DenseVectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.sample_docs = [
            {"id": "doc_paged.md", "content": "PagedAttention eliminates KV cache memory fragmentation using virtual paging."},
            {"id": "doc_rrf.md", "content": "Reciprocal Rank Fusion combines sparse BM25 and dense embeddings with constant k 60."},
            {"id": "doc_quant.md", "content": "Quantization FP8 and AWQ protect salient weights to compress models."}
        ]

    def test_bm25_retrieval(self):
        bm25 = BM25Retriever()
        bm25.index_documents(self.sample_docs)
        results = bm25.retrieve("PagedAttention memory fragmentation", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc_paged.md")

        # Test empty query and high threshold filtering
        self.assertEqual(bm25.retrieve(""), [])
        self.assertEqual(bm25.retrieve("irrelevant query", min_score=999.0), [])

    def test_dense_vector_retrieval(self):
        vstore = DenseVectorStore()
        vstore.index_documents(self.sample_docs)
        results = vstore.retrieve("virtual paging memory", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("score", results[0])

    def test_hybrid_rrf_retrieval(self):
        bm25 = BM25Retriever()
        bm25.index_documents(self.sample_docs)
        vstore = DenseVectorStore()
        vstore.index_documents(self.sample_docs)

        hybrid = HybridRetriever(bm25, vstore, rrf_k=60)
        results = hybrid.retrieve("Reciprocal Rank Fusion constant k", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc_rrf.md")

        # Test empty query and thresholding
        self.assertEqual(hybrid.retrieve(""), [])
        self.assertEqual(hybrid.retrieve("Reciprocal Rank Fusion", min_score=1.5), [])

    def test_cross_encoder_reranker(self):
        reranker = CrossEncoderReranker()
        candidates = [
            {"rank": 1, "doc_id": "doc_quant.md", "content": self.sample_docs[2]["content"], "score": 0.8},
            {"rank": 2, "doc_id": "doc_rrf.md", "content": self.sample_docs[1]["content"], "score": 0.5},
        ]
        reranked = reranker.rerank("Reciprocal Rank Fusion BM25", candidates, top_k=2)
        self.assertEqual(reranked[0]["doc_id"], "doc_rrf.md")

if __name__ == "__main__":
    unittest.main()
