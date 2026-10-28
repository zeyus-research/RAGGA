from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragga.pipeline.retrieval import BM25Index, HybridRetriever, tokenize
from ragga.pipeline.vectorstore import NumpyVectorStore


def test_tokenize_splits_camel_and_snake_case():
    assert tokenize("RiseTogether") == ["risetogether", "rise", "together"]
    assert tokenize("lab_streaming EEG2 Højbo") == ["lab_streaming", "lab", "streaming", "eeg2", "eeg", "2", "højbo"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_bm25_ranks_rare_term_first():
    index = BM25Index(["the cat sat", "the dog sat", "RiseTogether paradigm notes", "the the the"])
    assert index.search("tell me about RiseTogether", k=3) == [2]
    assert index.search("the sat", k=2)[0] in (0, 1)
    assert index.search("unknown", k=3) == []
    assert BM25Index([]).search("x", k=3) == []


def test_bm25_prefers_shorter_documents():
    index = BM25Index(["sound", "sound " + "filler " * 50])
    assert index.search("sound", k=2) == [0, 1]


def make_store(texts):
    return NumpyVectorStore.from_texts(
        texts, DeterministicFakeEmbedding(size=16), ids=[str(i) for i in range(len(texts))]
    )


def test_hybrid_fuses_and_limits():
    texts = [f"filler note {i}" for i in range(20)] + ["RiseTogether joint action game"]
    retriever = HybridRetriever(store=make_store(texts), k=3, fetch_k=10)
    hits = retriever.invoke("RiseTogether")
    assert len(hits) == 3
    assert hits[0].page_content == "RiseTogether joint action game"
    assert len({h.id for h in hits}) == 3


def test_hybrid_rebuilds_keyword_index_after_changes():
    store = make_store(["alpha", "beta"])
    retriever = HybridRetriever(store=store, k=1, fetch_k=5)
    assert retriever.invoke("alpha")[0].page_content == "alpha"
    store.add_texts(["gamma words"], ids=["2"])
    retriever.keyword_weight = 10.0  # make the keyword ranking decide
    assert retriever.invoke("gamma")[0].page_content == "gamma words"


def test_hybrid_empty_store():
    assert HybridRetriever(store=NumpyVectorStore(DeterministicFakeEmbedding(size=4))).invoke("x") == []


def test_documents_and_revision():
    store = make_store(["a", "b"])
    revision = store.revision
    assert [d.page_content for d in store.documents] == ["a", "b"]
    store.delete(["0"])
    assert store.revision > revision
    assert [d.page_content for d in store.documents] == ["b"]
    assert isinstance(store.documents[0], Document)


def test_hybrid_limits_chunks_per_source():
    texts = [f"RiseTogether part {i}" for i in range(6)] + ["RiseTogether elsewhere"]
    store = NumpyVectorStore.from_texts(
        texts,
        DeterministicFakeEmbedding(size=16),
        metadatas=[{"source": "big.md"}] * 6 + [{"source": "small.md"}],
        ids=[str(i) for i in range(7)],
    )
    hits = HybridRetriever(store=store, k=4, max_per_source=2).invoke("RiseTogether")
    assert [h.metadata["source"] for h in hits].count("big.md") == 2
    assert "small.md" in [h.metadata["source"] for h in hits]
    assert len(hits) == 3
