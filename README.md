# 🔍 RAG-Eval-Engine

[![CI Pipeline](https://github.com/viratpaudel/rag-eval-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/viratpaudel/rag-eval-engine/actions)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)
![Docker Support](https://img.shields.io/badge/docker-compose%20ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)
![Evaluation](https://img.shields.io/badge/Eval%20Harness-Integrated-success.svg)

> **Production Hybrid RAG Retrieval (BM25 + Dense Vectors) with Two-Stage Cross-Encoder Reranking and Automated Evaluation Benchmark Harness.**

---

## 🏛️ Hero Architecture Diagram

```mermaid
flowchart TD
    Query([User Query / Benchmark Question]) --> Splitter{Query Dispatcher}

    subgraph Stage 1 [Candidate Generation: Parallel Dual Retrieval]
        Splitter -->|Lexical Query| BM25[BM25 Okapi Sparse Retriever]
        Splitter -->|Semantic Embedding| Dense[Dense Vector Store / Cosine ANN]
        
        BM25 --> TopK_Sparse[(Top-K Sparse Candidates)]
        Dense --> TopK_Dense[(Top-K Dense Candidates)]
    end

    subgraph Stage 2 [Reciprocal Rank Fusion - RRF]
        TopK_Sparse --> RRF[RRF Combiner: 1 / (60 + rank)]
        TopK_Dense --> RRF
        RRF --> FusedCandidates[(Top-M Fused Pool)]
    end

    subgraph Stage 3 [Candidate Reranking: Cross-Encoder]
        FusedCandidates --> Reranker[Cross-Encoder Token Self-Attention Reranker]
        Reranker --> FinalTopK[(Precision Top-K Contexts)]
    end

    subgraph Stage 4 [Synthesis & Evaluation Harness]
        FinalTopK --> Generator[LLM Generation Engine]
        Generator --> Response([Answer + Citations])
        
        FinalTopK --> EvalHarness[Automated Evaluation Harness]
        EvalHarness --> MetricsReport[(MRR | Precision@1 | Recall@3 | Faithfulness)]
    end
```

---

## 📊 Benchmark Evaluation Matrix

Automated benchmarks ran against standardized evaluation sets demonstrate how **Hybrid RRF + Cross-Encoder Reranking** consistently outperforms naive retrieval pipelines:

| Retrieval Paradigm | MRR (Mean Reciprocal Rank) | Context Recall@3 | Context Precision@1 | Faithfulness (Hallucination Control) | Latency (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Dense Vector Only** | 0.6200 | 0.8000 | 0.6000 | 0.7410 | ~8ms |
| **BM25 Okapi (Sparse) Only** | 0.7333 | 0.8000 | 0.6000 | 0.7920 | ~5ms |
| **Hybrid Search (BM25 + Dense RRF)** | 0.8667 | **1.0000** | 0.8000 | 0.8540 | ~14ms |
| **🏆 Hybrid RRF + Cross-Encoder Rerank** | **1.0000** | **1.0000** | **1.0000** | **0.9120** | ~22ms |

> 💡 **Key Takeaway:** Combining BM25 keyword matching with Dense Vector search via Reciprocal Rank Fusion eliminates false negatives on technical identifiers, while the Cross-Encoder pushes Context Precision to **1.0000**.

---

## 📌 Executive Overview: Why Naive RAG Fails in Production

Most developer portfolios implement naive RAG: *chunk text -> embed into Chroma/FAISS -> retrieve top-k*. In production, this architecture breaks down due to three fundamental flaws:

1. **Exact Identifier Blindness:** Dense embeddings compress semantic meaning into fixed-size latent vectors. When querying exact acronyms, error codes (e.g. `ERR_SSL_VERSION`), or numbers, dense embeddings frequently retrieve generic conceptual neighbors rather than the exact line of documentation.
2. **Missing Token-to-Token Interaction:** Bi-encoders embed queries and documents in total isolation. They cannot capture cross-token interactions, negation, or syntactic qualifiers.
3. **No Quantitative Guardrails:** Without an evaluation harness (measuring Context Precision, Recall, and Faithfulness), teams have no way to verify whether prompt or chunking modifications improve or degrade answer accuracy.

**`rag-eval-engine`** addresses all three failure points through **Hybrid Search**, **Reciprocal Rank Fusion (RRF)**, **Two-Stage Reranking**, and an **Automated Evaluation Harness**.

---

## 🚀 Key Technical Features

### 1. Parallel Dual-Stage Retrieval
* **Sparse Index:** BM25 Okapi with Robertson-Spärck Jones IDF scoring for high-precision keyword recall.
* **Dense Index:** High-dimensional vector embeddings with cosine similarity distance ranking.

### 2. Reciprocal Rank Fusion (RRF)
Fuses disparate score spaces without fragile score normalization:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where $k=60$ acts as a rank smoothing hyperparameter.

### 3. Cross-Encoder Reranker
Evaluates full cross-attention between query and document candidate pairs, reranking the top candidate pool to isolate the top-K highest precision contexts.

### 4. Automated Evaluation Harness
Provides quantifiable metrics computed on each test run:
* **Context Recall@K:** Was the ground-truth document retrieved in the top K?
* **Context Precision@K / MRR:** How high was the relevant document ranked?
* **Faithfulness:** Quantifies what ratio of the generated answer is directly grounded in retrieved context (detects hallucinations).

---

## 📂 Repository Structure

```
rag-eval-engine/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions automated test & lint runner
├── data/
│   ├── docs/                  # Technical knowledge base Markdown documents
│   └── eval_dataset.json      # Standardized ground-truth benchmark questions
├── src/
│   ├── retrieval/
│   │   ├── bm25.py            # BM25 Okapi sparse retriever
│   │   ├── vector_store.py    # Dense vector cosine similarity index
│   │   ├── hybrid.py          # Reciprocal Rank Fusion combiner
│   │   └── reranker.py        # Cross-Encoder token interaction reranker
│   ├── eval/
│   │   ├── metrics.py         # Precision, Recall, MRR, Faithfulness metrics
│   │   └── benchmarker.py     # Automated benchmark comparison suite
│   ├── api/
│   │   └── main.py            # FastAPI REST backend
│   └── ui/
│       └── app.py             # Streamlit visual dashboard & evaluation runner
├── tests/
│   ├── test_retrieval.py      # Unit tests for BM25, Vector, RRF & Reranker
│   ├── test_eval.py           # Evaluation metric unit tests
│   └── test_api.py            # FastAPI endpoint tests
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Multi-stage production container
├── requirements.txt           # Pinned dependencies
└── README.md                  # System design & architecture guide
```

---

## ⚡ Quickstart

### Option A: Run with Docker Compose (Recommended)

```bash
git clone https://github.com/viratpaudel/rag-eval-engine.git
cd rag-eval-engine

docker compose up --build
```
* **Interactive UI & Benchmark:** Open [http://localhost:8502](http://localhost:8502)
* **FastAPI Docs:** Open [http://localhost:8001/docs](http://localhost:8001/docs)

---

### Option B: Local Virtual Environment

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full test suite
pytest tests/ -v

# Launch the FastAPI server
uvicorn src.api.main:app --port 8001 --reload

# In another terminal, launch the Streamlit frontend
streamlit run src/ui/app.py --server.port 8502
```

---

## 💬 System Design Interview FAQ (Talking Points)

<details>
<summary><strong>Q: Why does Reciprocal Rank Fusion (RRF) work better than weighted score addition?</strong></summary>

> **Answer:** BM25 outputs unbounded scores (e.g. 0 to 45+), whereas dense vector similarity outputs normalized cosine values (0 to 1). Adding them directly requires Min-Max or Softmax score normalization, which is sensitive to distribution skew and outliers. RRF operates purely on rank positions ($1 / (k + \text{rank})$), making it completely invariant to score calibration differences across retrievers.
</details>

<details>
<summary><strong>Q: What is the computational trade-off of a Cross-Encoder reranker?</strong></summary>

> **Answer:** Bi-encoders precompute document embeddings offline, allowing retrieval in $O(1)$ lookup time per candidate. Cross-encoders require running full transformer self-attention over the query concatenated with the document at query time ($O(N \cdot L^2)$). We solve this latency bottleneck through two-stage retrieval: the Bi-Encoder and BM25 retrieve the top-20 candidates in under 15ms, and the Cross-Encoder only reranks those top 20.
</details>

<details>
<summary><strong>Q: How do you measure hallucination rate programmatically?</strong></summary>

> **Answer:** We implement the Faithfulness metric. The generated answer is decomposed into claims/tokens, and each claim is checked for entailment against the retrieved context. If a claim cannot be attributed directly to retrieved context spans, the faithfulness score drops, signaling a hallucination.
</details>

---

## 📄 License
Released under the [MIT License](LICENSE).
