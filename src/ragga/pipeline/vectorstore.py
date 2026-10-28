"""A small exact (brute-force) vector store backed by a numpy matrix.

For personal-notes sized collections (up to a few hundred thousand chunks) a single matrix-vector product is as fast
as FAISS' flat index, and the store persists as plain `.npy` + JSON instead of pickle.
"""

import json
import os
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

VECTORS_FILE = "vectors.npy"
DOCUMENTS_FILE = "documents.json"


class NumpyVectorStore(VectorStore):
    """Exact nearest-neighbour search by L2 distance (same ranking as FAISS `IndexFlatL2`)."""

    def __init__(self, embedding: Embeddings) -> None:
        self._embedding = embedding
        self._ids: list[str] = []
        self._docs: dict[str, Document] = {}
        self._vectors: np.ndarray | None = None  # (n, dim) float32, row i belongs to self._ids[i]
        self._sq_norms: np.ndarray | None = None  # squared row norms, so a search is one matrix-vector product

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def __len__(self) -> int:
        return len(self._ids)

    def _set_vectors(self, vectors: np.ndarray | None) -> None:
        self._vectors = vectors
        self._sq_norms = None if vectors is None else np.einsum("ij,ij->i", vectors, vectors)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        ids = [i or str(uuid.uuid4()) for i in ids] if ids else [str(uuid.uuid4()) for _ in texts]
        if len(metadatas) != len(texts) or len(ids) != len(texts):
            msg = "texts, metadatas and ids must have the same length"
            raise ValueError(msg)
        self.delete([i for i in ids if i in self._docs])  # re-adding an id replaces it

        new = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        self._set_vectors(new if self._vectors is None else np.vstack([self._vectors, new]))
        for doc_id, text, metadata in zip(ids, texts, metadatas, strict=True):
            self._ids.append(doc_id)
            self._docs[doc_id] = Document(id=doc_id, page_content=text, metadata=metadata)
        return ids

    def delete(self, ids: list[str] | None = None, **_kwargs: Any) -> bool | None:
        if ids is None:
            self._ids, self._docs = [], {}
            self._set_vectors(None)
            return True
        remove = set(ids) & self._docs.keys()
        if not remove:
            return True
        keep = [i for i, doc_id in enumerate(self._ids) if doc_id not in remove]
        self._ids = [self._ids[i] for i in keep]
        for doc_id in remove:
            del self._docs[doc_id]
        self._set_vectors(self._vectors[keep] if keep and self._vectors is not None else None)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self._docs[i] for i in ids if i in self._docs]

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4
    ) -> list[tuple[Document, float]]:
        if self._vectors is None or self._sq_norms is None or k <= 0:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        # squared L2 distance, minus the constant |q|^2 term which doesn't change the ranking
        dist = self._sq_norms - 2 * (self._vectors @ query)
        k = min(k, len(dist))
        top = np.argpartition(dist, k - 1)[:k]
        top = top[np.argsort(dist[top])]
        q_sq = float(query @ query)
        return [(self._docs[self._ids[i]], float(np.sqrt(max(dist[i] + q_sq, 0.0)))) for i in top]

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **_kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score_by_vector(embedding, k)]

    def similarity_search_with_score(self, query: str, k: int = 4, **_kwargs: Any) -> list[tuple[Document, float]]:
        return self.similarity_search_with_score_by_vector(self._embedding.embed_query(query), k)

    def similarity_search(self, query: str, k: int = 4, **_kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._euclidean_relevance_score_fn

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> Self:
        store = cls(embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def save(self, folder: str | Path) -> None:
        """Write the store to `folder` (vectors as .npy, documents as JSON). Each file is replaced atomically."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.empty((0, 0), dtype=np.float32)
        docs = [
            {"id": i, "page_content": self._docs[i].page_content, "metadata": self._docs[i].metadata} for i in self._ids
        ]
        tmp_vectors, tmp_docs = folder / f".{VECTORS_FILE}.tmp", folder / f".{DOCUMENTS_FILE}.tmp"
        with open(tmp_vectors, "wb") as f:
            np.save(f, vectors, allow_pickle=False)
        with open(tmp_docs, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, default=str)
        os.replace(tmp_vectors, folder / VECTORS_FILE)
        os.replace(tmp_docs, folder / DOCUMENTS_FILE)

    @classmethod
    def exists(cls, folder: str | Path) -> bool:
        return (Path(folder) / VECTORS_FILE).is_file() and (Path(folder) / DOCUMENTS_FILE).is_file()

    @classmethod
    def load(cls, folder: str | Path, embedding: Embeddings) -> Self:
        folder = Path(folder)
        store = cls(embedding)
        vectors = np.load(folder / VECTORS_FILE, allow_pickle=False)
        with open(folder / DOCUMENTS_FILE, encoding="utf-8") as f:
            docs = json.load(f)
        if len(docs) != (vectors.shape[0] if vectors.size else 0):
            msg = f"Corrupt vector store in {folder}: {len(docs)} documents but {vectors.shape[0]} vectors"
            raise ValueError(msg)
        for d in docs:
            store._ids.append(d["id"])
            store._docs[d["id"]] = Document(id=d["id"], page_content=d["page_content"], metadata=d["metadata"])
        store._set_vectors(vectors.astype(np.float32, copy=False) if docs else None)
        return store
