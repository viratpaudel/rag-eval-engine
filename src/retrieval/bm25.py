import math
import re
from typing import Dict, List, Tuple
from collections import Counter

class BM25Retriever:
    """Production BM25 Okapi lexical sparse retrieval implementation."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Counter] = []
        self.total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Sanitizes and tokenizes text into normalized lowercase tokens."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
        return tokens

    def index_documents(self, documents: List[Dict[str, str]]):
        """Indexes a list of documents with 'id' and 'content'."""
        self.doc_ids = []
        self.doc_contents = []
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            tokens = self._tokenize(content)
            
            self.doc_ids.append(doc_id)
            self.doc_contents.append(content)
            self.doc_lens.append(len(tokens))

            tf = Counter(tokens)
            self.term_freqs.append(tf)

            for token in tf.keys():
                self.doc_freqs[token] += 1

        self.total_docs = len(self.doc_ids)
        self.avg_dl = sum(self.doc_lens) / max(1, self.total_docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """Scores documents for a query using BM25 Okapi formula."""
        query_tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []

        for idx in range(self.total_docs):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf_counter = self.term_freqs[idx]

            for term in query_tokens:
                if term in tf_counter:
                    tf = tf_counter[term]
                    df = self.doc_freqs.get(term, 0)
                    # IDF with Robertson-Spärck Jones formula
                    idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    
                    # BM25 TF component
                    tf_norm = (tf * (self.k1 + 1.0)) / (
                        tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_dl)))
                    )
                    score += idf * tf_norm

            scores.append((idx, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, sc) in enumerate(scores[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": self.doc_ids[idx],
                "score": round(sc, 4),
                "content": self.doc_contents[idx],
                "retriever": "BM25_Okapi"
            })
        return results
