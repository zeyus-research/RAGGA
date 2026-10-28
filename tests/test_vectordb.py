from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragga import VectorDatabase


@pytest.fixture
def open_db(config, fake_encoder):
    """Create VectorDatabases that are closed after the test."""
    dbs: list[VectorDatabase] = []

    def factory() -> VectorDatabase:
        dbs.append(VectorDatabase(config, fake_encoder))  # type: ignore[arg-type]
        return dbs[-1]

    yield factory
    for db in dbs:
        db.close()


def make_docs() -> list[Document]:
    return [
        Document(page_content="Sonification turns data into sound.", metadata={"source": "a.md"}),
        Document(page_content="Faiss is a library for vector search.", metadata={"source": "b.md"}),
    ]


def test_empty_db_has_no_placeholder(open_db):
    db = open_db()
    assert db.db.similarity_search("anything", k=4) == []


def test_index_search_and_reload(open_db):
    db = open_db()
    assert not db.loaded_from_disk
    db.documents = make_docs()

    query = "Sonification turns data into sound."
    hits = db.db.similarity_search(query, k=1)
    assert hits[0].page_content == query
    assert query in db.get_similar_docs(query, k=1)
    assert len(db.retriever.invoke(query)) == 2

    reloaded = open_db()
    assert reloaded.loaded_from_disk
    assert reloaded.db.similarity_search(query, k=1)[0].page_content == query


def test_reindexing_same_docs_does_not_duplicate(open_db):
    db = open_db()
    db.documents = make_docs()
    _ = db.db
    db.documents = make_docs()
    _ = db.db
    assert len(db.db.similarity_search("x", k=10)) == 2


def test_merge_documents(open_db):
    db = open_db()
    db.documents = make_docs()[:1]
    db.merge_documents(make_docs()[1:])
    assert len(db.db.similarity_search("x", k=10)) == 2


def test_index_from_older_version_is_rebuilt(open_db, config):
    # Simulate a cache left by an older RAGGA (or a deleted vector dir): the record manager still lists the docs as
    # indexed, but there are no stored vectors. They must be re-indexed instead of silently skipped.
    db = open_db()
    db.documents = make_docs()
    _ = db.db
    for f in (Path(config["retriever"]["cache_dir"]) / ".vectors").iterdir():
        f.unlink()
    rebuilt = open_db()
    rebuilt.documents = make_docs()
    assert len(rebuilt.db.similarity_search("x", k=10)) == 2
