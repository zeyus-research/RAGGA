import json

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragga.pipeline.vectorstore import DOCUMENTS_FILE, VECTORS_FILE, NumpyVectorStore

TEXTS = [f"note number {i} about topic {i % 7}" for i in range(50)]


@pytest.fixture
def store() -> NumpyVectorStore:
    return NumpyVectorStore.from_texts(
        TEXTS,
        DeterministicFakeEmbedding(size=16),
        metadatas=[{"n": i} for i in range(50)],
        ids=[str(i) for i in range(50)],
    )


def test_matches_brute_force_l2(store):
    emb = store.embeddings
    vectors = np.asarray(emb.embed_documents(TEXTS))
    query = np.asarray(emb.embed_query("query"))
    expected = np.argsort(np.linalg.norm(vectors - query, axis=1))[:5]
    results = store.similarity_search_with_score("query", k=5)
    assert [int(doc.id) for doc, _ in results] == expected.tolist()
    assert [score for _, score in results] == pytest.approx(np.linalg.norm(vectors - query, axis=1)[expected], rel=1e-4)
    assert results[0][0].metadata == {"n": int(expected[0])}


def test_exact_match_ranks_first(store):
    assert store.similarity_search(TEXTS[17], k=1)[0].id == "17"


def test_k_larger_than_store_and_empty_store():
    empty = NumpyVectorStore(DeterministicFakeEmbedding(size=8))
    assert empty.similarity_search("x") == []
    empty.add_texts(["only one"])
    assert len(empty.similarity_search("x", k=10)) == 1


def test_delete_and_replace(store):
    store.delete(["3", "4", "missing"])
    assert len(store) == 48
    assert store.get_by_ids(["3", "5"])[0].id == "5"
    assert all(doc.id not in {"3", "4"} for doc in store.similarity_search("note", k=50))
    store.add_texts(["replaced text"], ids=["5"])  # same id replaces, doesn't duplicate
    assert len(store) == 48
    assert store.get_by_ids(["5"])[0].page_content == "replaced text"
    assert store.similarity_search("replaced text", k=1)[0].id == "5"
    store.delete()
    assert len(store) == 0
    assert store.similarity_search("note") == []


def test_save_and_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    assert NumpyVectorStore.exists(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path, store.embeddings)
    assert len(loaded) == len(store)
    assert loaded.similarity_search_with_score("query", k=5) == store.similarity_search_with_score("query", k=5)
    # the documents file is plain JSON, and vectors load with pickle disabled
    assert json.loads((tmp_path / DOCUMENTS_FILE).read_text())[0]["metadata"] == {"n": 0}
    np.load(tmp_path / VECTORS_FILE, allow_pickle=False)


def test_save_empty_store(tmp_path):
    NumpyVectorStore(DeterministicFakeEmbedding(size=8)).save(tmp_path)
    assert len(NumpyVectorStore.load(tmp_path, DeterministicFakeEmbedding(size=8))) == 0


def test_load_rejects_mismatched_files(store, tmp_path):
    store.save(tmp_path)
    (tmp_path / DOCUMENTS_FILE).write_text("[]")
    with pytest.raises(ValueError, match="Corrupt vector store"):
        NumpyVectorStore.load(tmp_path, store.embeddings)
