import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.eval.benchmarker import benchmarker

app = FastAPI(
    title="RAG-Eval-Engine API",
    description="Production Hybrid RAG Retrieval (BM25 + Dense) with Automated Evaluation Harness",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrievalRequest(BaseModel):
    query: str = Field(..., example="How does PagedAttention eliminate memory fragmentation?")
    strategy: str = Field("hybrid_rerank", description="One of: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'")
    top_k: int = Field(3, ge=1, le=10)

class GenerationRequest(BaseModel):
    query: str
    strategy: str = "hybrid_rerank"
    top_k: int = 3

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "rag-eval-engine",
        "indexed_documents": len(benchmarker.documents),
        "supported_strategies": ["dense", "bm25", "hybrid", "hybrid_rerank"],
    }

@app.post("/retrieve")
def retrieve_documents(req: RetrievalRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    strategy = req.strategy.lower()
    if strategy == "dense":
        results = benchmarker.vector_store.retrieve(q, top_k=req.top_k)
    elif strategy == "bm25":
        results = benchmarker.bm25.retrieve(q, top_k=req.top_k)
    elif strategy == "hybrid":
        results = benchmarker.hybrid.retrieve(q, top_k=req.top_k)
    elif strategy in ("hybrid_rerank", "hybrid_reranked"):
        first_stage = benchmarker.hybrid.retrieve(q, top_k=req.top_k * 2)
        results = benchmarker.reranker.rerank(q, first_stage, top_k=req.top_k)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy: {req.strategy}")

    return {
        "query": q,
        "strategy": req.strategy,
        "results_count": len(results),
        "documents": results,
    }

@app.post("/generate")
def generate_rag_answer(req: GenerationRequest):
    retrieval_res = retrieve_documents(RetrievalRequest(query=req.query, strategy=req.strategy, top_k=req.top_k))
    docs = retrieval_res["documents"]

    if not docs:
        return {"answer": "No relevant documents found to answer query.", "citations": []}

    # Context assembly
    contexts = [f"[{d['doc_id']}] {d['content'][:300]}..." for d in docs]
    citations = [d["doc_id"] for d in docs]

    # Deterministic synthesis / Gemini LLM synthesis if key available
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            prompt = f"Context:\n{' '.join(contexts)}\n\nQuery: {req.query}\nProvide a direct answer with citations."
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            answer = resp.text or ""
        except Exception:
            answer = f"Synthesized answer based on {len(docs)} retrieved sources: {docs[0]['content'][:200]}..."
    else:
        answer = f"Based on retrieved documentation ({', '.join(citations)}):\n\n{docs[0]['content'][:350]}..."

    return {
        "query": req.query,
        "strategy": req.strategy,
        "answer": answer,
        "citations": citations,
        "retrieved_count": len(docs),
    }

@app.get("/benchmark")
def run_evaluation_benchmark():
    """Runs automated evaluation suite comparing all 4 retrieval paradigms."""
    results = benchmarker.run_benchmark()
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)
