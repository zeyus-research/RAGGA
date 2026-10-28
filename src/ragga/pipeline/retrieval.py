"""Hybrid retrieval: BM25 keyword search plus vector search, merged with reciprocal rank fusion.

Embeddings are good at meaning ("joint action game" finds "collaborative paradigm") but weak on rare names and
jargon, which is exactly what personal notes are full of ("RiseTogether", "liblsl"). BM25 is the opposite, so using
both finds notes either one alone would miss.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from ragga.pipeline.vectorstore import NumpyVectorStore

_WORD = re.compile(r"\w+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, plus the parts of CamelCase and snake_case words (RiseTogether -> rise, together)."""
    tokens = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        # _CAMEL only knows ASCII letters, so leave e.g. "Højbo" whole
        parts = [
            p.lower() for piece in word.split("_") for p in (_CAMEL.findall(piece) if piece.isascii() else [piece])
        ]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed list of texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._n = len(texts)
        lengths = np.zeros(self._n, dtype=np.float32)
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings[term][0].append(i)
                postings[term][1].append(tf)
        self._norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()) if self._n else 0.0, 1.0))
        self._postings = {
            term: (np.asarray(docs, dtype=np.int64), np.asarray(tfs, dtype=np.float32))
            for term, (docs, tfs) in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self._n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            docs, tfs = self._postings[term]
            idf = math.log(1 + (self._n - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += idf * tfs * (self.k1 + 1) / (tfs + self._norm[docs])
        return scores

    def search(self, query: str, k: int) -> list[int]:
        """Indices of the top `k` texts, best first (texts sharing no term with the query are left out)."""
        scores = self.scores(query)
        k = min(k, int(np.count_nonzero(scores)))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()


class HybridRetriever(BaseRetriever):
    """Retrieve from a `NumpyVectorStore` with both vector and BM25 search, fused by reciprocal rank."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: NumpyVectorStore
    k: int = 4
    fetch_k: int = 50  # candidates taken from each search before fusing
    rrf_k: int = 60  # standard RRF constant: higher values flatten the difference between ranks
    keyword_weight: float = 1.0  # weight of the BM25 ranking relative to the vector ranking
    max_per_source: int | None = None  # at most this many chunks per note, so one long note can't crowd out the rest

    _bm25: BM25Index | None = PrivateAttr(default=None)
    _bm25_revision: int = PrivateAttr(default=-1)

    def _keyword_index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_revision != self.store.revision:
            self._bm25 = BM25Index([d.page_content for d in self.store.documents])
            self._bm25_revision = self.store.revision
        return self._bm25

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        if len(self.store) == 0:
            return []
        docs = self.store.documents
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, Document] = {}
        for rank, doc in enumerate(self.store.similarity_search(query, k=self.fetch_k)):
            fused[doc.id] += 1 / (self.rrf_k + rank + 1)  # type: ignore[index]
            by_id[doc.id] = doc  # type: ignore[index]
        for rank, i in enumerate(self._keyword_index().search(query, self.fetch_k)):
            fused[docs[i].id] += self.keyword_weight / (self.rrf_k + rank + 1)  # type: ignore[index]
            by_id[docs[i].id] = docs[i]  # type: ignore[index]
        results: list[Document] = []
        per_source: Counter[str] = Counter()
        for doc_id in sorted(fused, key=fused.__getitem__, reverse=True):
            source = by_id[doc_id].metadata.get("source", doc_id)
            if self.max_per_source is not None and per_source[source] >= self.max_per_source:
                continue
            per_source[source] += 1
            results.append(by_id[doc_id])
            if len(results) == self.k:
                break
        return results
