# Cross-Encoder Rerankers & Two-Stage Retrieval

Production Information Retrieval architectures separate retrieval into two stages:
1. Candidate Generation (Bi-Encoder Retrieval): Fast first-stage retrieval over millions of documents fetching top-100 candidates in under 15ms.
2. Candidate Reranking (Cross-Encoder): Deep transformer-based scoring over the top-50 candidates to output precise top-5 contexts for generation.

### Bi-Encoders vs. Cross-Encoders
* **Bi-Encoders (Dense Embeddings):** Embed the query and document independently into vectors $u = f(q)$ and $v = g(d)$. Similarity is a dot product $\langle u, v \rangle$. Because vectors can be precomputed, retrieval is extremely fast, but cross-attention between query tokens and document tokens is completely lost.
* **Cross-Encoders:** Feed the query and document concatenated together directly through all self-attention layers: $\text{score} = \text{Transformer}([CLS] \circ \text{query} \circ [SEP] \circ \text{document})$. This allows full token-to-token cross-attention, accurately detecting negation, nuance, and structural dependencies that bi-encoders miss.

Reranking the top-20 retrieved candidates with a Cross-Encoder typically improves Context Precision and NDCG@10 by 15-35%.
