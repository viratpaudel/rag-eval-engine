import streamlit as st
import pandas as pd
from pathlib import Path

from src.eval.benchmarker import benchmarker

st.set_page_config(
    page_title="RAG-Eval-Engine | Hybrid Search & Benchmark Harness",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG-Eval-Engine")
st.caption("Production Hybrid Retrieval (BM25 + Dense Vectors) with Cross-Encoder Reranking & Automated Eval Harness")

with st.sidebar:
    st.header("⚙️ Engine Configuration")
    st.info("Demonstrating the limitations of Naive Vector Search vs. Hybrid RRF and Two-Stage Cross-Encoder Reranking.")
    
    st.markdown("### 📚 Indexed Documents")
    for doc in benchmarker.documents:
        with st.expander(f"📄 `{doc['id']}`"):
            st.caption(f"Length: {len(doc['content'].split())} words")
            st.text(doc['content'][:180] + "...")

    st.divider()
    st.markdown("**Key Architectural Concepts:**")
    st.markdown("- Reciprocal Rank Fusion ($k=60$)")
    st.markdown("- Cross-Attention Reranking")
    st.markdown("- Automated Metrics: MRR, Recall@3, Precision@1, Faithfulness")

tab_search, tab_benchmark = st.tabs(["🔎 Interactive Query & Retrieval", "📊 Automated Evaluation Benchmark"])

with tab_search:
    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Search or ask a technical question:",
            value="How does PagedAttention eliminate memory fragmentation in KV cache?",
            help="Compare different retrieval paradigms on exact technical keywords vs conceptual questions."
        )
    with col2:
        strategy = st.selectbox(
            "Retrieval Strategy:",
            ["Hybrid_Reranked", "Hybrid_RRF", "Dense_Only", "BM25_Only"],
            index=0
        )

    col_btn, _ = st.columns([1, 4])
    search_clicked = col_btn.button("Run Retrieval", type="primary")

    if search_clicked or query_input:
        if strategy == "Dense_Only":
            results = benchmarker.vector_store.retrieve(query_input, top_k=3)
        elif strategy == "BM25_Only":
            results = benchmarker.bm25.retrieve(query_input, top_k=3)
        elif strategy == "Hybrid_RRF":
            results = benchmarker.hybrid.retrieve(query_input, top_k=3)
        else:
            candidates = benchmarker.hybrid.retrieve(query_input, top_k=6)
            results = benchmarker.reranker.rerank(query_input, candidates, top_k=3)

        st.subheader(f"Top Retrieved Contexts (Strategy: `{strategy}`)")
        for idx, res in enumerate(results, start=1):
            with st.container():
                st.markdown(f"**#{idx} [{res['doc_id']}]** | Score: `{res['score']}` | Paradigm: `{res['retriever']}`")
                st.info(res["content"][:400] + "...")

with tab_benchmark:
    st.subheader("Automated Retrieval Evaluation Harness")
    st.write("Runs the standardized evaluation test set across all 4 retrieval strategies and calculates quantitative IR metrics.")

    if st.button("🚀 Execute Full Benchmark Suite", type="secondary"):
        with st.spinner("Benchmarking Dense, BM25, Hybrid RRF, and Reranked pipelines..."):
            bench_results = benchmarker.run_benchmark()

        df_summary = pd.DataFrame(bench_results["summary_table"])
        st.dataframe(df_summary, use_container_width=True)

        st.success(f"Benchmark completed successfully over {bench_results['total_test_queries']} test cases. Best Performing Pipeline: `{bench_results['best_strategy']}`.")
