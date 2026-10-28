from langchain_core.documents import Document

from ragga import VectorDatabase


def make_docs() -> list[Document]:
    return [
        Document(page_content="Sonification turns data into sound.", metadata={"source": "a.md"}),
        Document(page_content="Faiss is a library for vector search.", metadata={"source": "b.md"}),
    ]


def test_empty_db_has_no_placeholder(config, fake_encoder):
    db = VectorDatabase(config, fake_encoder)  # type: ignore[arg-type]
    assert db.db.similarity_search("anything", k=4) == []


def test_index_search_and_reload(config, fake_encoder):
    db = VectorDatabase(config, fake_encoder)  # type: ignore[arg-type]
    assert not db.loaded_from_disk
    db.documents = make_docs()

    query = "Sonification turns data into sound."
    hits = db.db.similarity_search(query, k=1)
    assert hits[0].page_content == query
    assert query in db.get_similar_docs(query, k=1)
    assert len(db.retriever.invoke(query)) == 2

    reloaded = VectorDatabase(config, fake_encoder)  # type: ignore[arg-type]
    assert reloaded.loaded_from_disk
    assert reloaded.db.similarity_search(query, k=1)[0].page_content == query


def test_reindexing_same_docs_does_not_duplicate(config, fake_encoder):
    db = VectorDatabase(config, fake_encoder)  # type: ignore[arg-type]
    db.documents = make_docs()
    _ = db.db
    db.documents = make_docs()
    _ = db.db
    assert len(db.db.similarity_search("x", k=10)) == 2


def test_merge_documents(config, fake_encoder):
    db = VectorDatabase(config, fake_encoder)  # type: ignore[arg-type]
    db.documents = make_docs()[:1]
    db.merge_documents(make_docs()[1:])
    assert len(db.db.similarity_search("x", k=10)) == 2
