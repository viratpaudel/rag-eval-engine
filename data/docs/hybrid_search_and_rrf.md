# Hybrid Search & Reciprocal Rank Fusion (RRF)

While dense vector embeddings excel at capturing high-level conceptual and semantic similarity, they consistently struggle with exact keyword matching, technical identifiers, error codes, SKUs, and rare proper nouns.

### The Limits of Naive Vector Search
If a user queries for an exact error code like `ERR_SSL_VERSION_OR_CIPHER_MISMATCH`, dense embeddings often map to generic SSL/TLS concepts rather than the exact documentation matching that literal code. Conversely, sparse keyword search (BM25) excels at token-exact retrieval but fails at semantic synonym matching.

### Reciprocal Rank Fusion (RRF)
Hybrid search executes sparse (BM25) and dense (vector embedding) searches in parallel, then combines candidate rankings using Reciprocal Rank Fusion without requiring score normalization:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
- $M$ is the set of rankers (Sparse and Dense)
- $r_m(d)$ is the 1-based rank position of document $d$ in system $m$
- $k$ is a smoothing constant (typically set to 60) to prevent outlier ranks from dominating

RRF ensures robust retrieval across both abstract conceptual queries and exact keyword searches.
