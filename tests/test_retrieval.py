import pytest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import DenseVectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

@pytest.fixture
def sample_docs():
    return [
        {"id": "doc_paged.md", "content": "PagedAttention eliminates KV cache memory fragmentation using virtual paging."},
        {"id": "doc_rrf.md", "content": "Reciprocal Rank Fusion combines sparse BM25 and dense embeddings with constant k 60."},
        {"id": "doc_quant.md", "content": "Quantization FP8 and AWQ protect salient weights to compress models."}
    ]

def test_bm25_retrieval(sample_docs):
    bm25 = BM25Retriever()
    bm25.index_documents(sample_docs)
    results = bm25.retrieve("PagedAttention memory fragmentation", top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_paged.md"

def test_dense_vector_retrieval(sample_docs):
    vstore = DenseVectorStore()
    vstore.index_documents(sample_docs)
    results = vstore.retrieve("virtual paging memory", top_k=2)
    assert len(results) == 2
    assert "score" in results[0]

def test_hybrid_rrf_retrieval(sample_docs):
    bm25 = BM25Retriever()
    bm25.index_documents(sample_docs)
    vstore = DenseVectorStore()
    vstore.index_documents(sample_docs)

    hybrid = HybridRetriever(bm25, vstore, rrf_k=60)
    results = hybrid.retrieve("Reciprocal Rank Fusion constant k", top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_rrf.md"

def test_cross_encoder_reranker(sample_docs):
    reranker = CrossEncoderReranker()
    candidates = [
        {"rank": 1, "doc_id": "doc_quant.md", "content": sample_docs[2]["content"], "score": 0.8},
        {"rank": 2, "doc_id": "doc_rrf.md", "content": sample_docs[1]["content"], "score": 0.5},
    ]
    reranked = reranker.rerank("Reciprocal Rank Fusion BM25", candidates, top_k=2)
    assert reranked[0]["doc_id"] == "doc_rrf.md"
