import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_retrieve():
    res = client.post("/retrieve", json={
        "query": "PagedAttention KV cache",
        "strategy": "hybrid_rerank",
        "top_k": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["results_count"] == 2
    assert len(data["documents"]) == 2

def test_api_generate():
    res = client.post("/generate", json={
        "query": "What is Reciprocal Rank Fusion?",
        "strategy": "hybrid_rerank",
        "top_k": 2
    })
    assert res.status_code == 200
    assert "answer" in res.json()
    assert len(res.json()["citations"]) >= 1

def test_api_benchmark_endpoint():
    res = client.get("/benchmark")
    assert res.status_code == 200
    assert "summary_table" in res.json()
