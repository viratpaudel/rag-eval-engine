# LLM Inference Optimization & KV Caching

Large Language Model (LLM) serving in production encounters memory bandwidth bottlenecks during autoregressive token generation. The primary technique to accelerate inference is the Key-Value (KV) cache, which stores the key and value projection tensors of previous tokens across attention layers.

### PagedAttention and vLLM
Traditional KV caches allocate contiguous physical memory blocks per sequence, leading to 60-80% memory waste due to internal and external memory fragmentation. PagedAttention divides the KV cache into fixed-size virtual memory blocks (e.g., 16 tokens per block) managed via a page table, mirroring virtual memory paging in operating systems. This achieves near-zero memory waste and enables continuous batching.

### Speculative Decoding
Speculative decoding utilizes a smaller, lightweight draft model (e.g., 1B parameters) to propose K candidate tokens in parallel, which are subsequently verified in a single forward pass by the primary target model (e.g., 70B parameters). If all draft tokens are accepted, inference speedups of 2x to 3x are achieved without altering output probability distributions.

### Quantization (FP8, INT4, AWQ)
Quantization reduces weight and activation precision. Activation-aware Weight Quantization (AWQ) identifies the top 1% salient weights based on activation magnitudes and protects them in higher precision, allowing 4-bit weight compression with negligible perplexity degradation.
