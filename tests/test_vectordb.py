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
    assert hits[0].page_content == query  # no title metadata, so no header
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
    for f in (Path(config["retriever"]["cache_dir"]) / ".vectors").glob("*/*"):
        f.unlink()
    rebuilt = open_db()
    rebuilt.documents = make_docs()
    assert len(rebuilt.db.similarity_search("x", k=10)) == 2


def make_notes() -> list[Document]:
    return [
        Document(
            page_content="A game paradigm for group joint action research.",
            metadata={"source": "data/PhD/RiseTogether.md", "title": "RiseTogether", "folder": "PhD"},
        ),
        Document(page_content="Bread needs flour and water.", metadata={"source": "data/bread.md", "title": "bread"}),
        Document(page_content="Notes about sound and data.", metadata={"source": "data/sound.md", "title": "sound"}),
    ]


def test_passages_carry_note_header(open_db):
    db = open_db()
    db.documents = make_notes()
    first = db.passages[0]
    assert first.page_content.startswith("Note: RiseTogether (PhD)\n\n")
    assert first.metadata["header"] == "Note: RiseTogether (PhD)\n\n"
    assert first.metadata["start_index"] == 0
    assert db.passages[1].page_content.startswith("Note: bread\n\n")


def test_hybrid_retriever_finds_note_by_name(open_db):
    db = open_db()
    db.documents = make_notes()
    hits = db.retriever.invoke("Tell me about Rise Together")
    assert hits[0].metadata["title"] == "RiseTogether"


def test_vector_search_option_and_bad_option(open_db, config):
    config["retriever"]["search"] = "vector"
    db = open_db()
    db.documents = make_notes()
    assert len(db.retriever.invoke("anything")) == 2
    config["retriever"]["search"] = "nope"
    with pytest.raises(ValueError, match=r"retriever\.search"):
        _ = open_db().retriever


def test_index_is_per_model_and_chunking(open_db, config, fake_encoder):
    db = open_db()
    db.documents = make_docs()
    _ = db.db
    config["retriever"]["splitting"]["chunk_size"] = 128
    rechunked = open_db()
    assert not rechunked.loaded_from_disk
    config["retriever"]["splitting"]["chunk_size"] = 64
    fake_encoder.model_name = "other"
    assert not open_db().loaded_from_disk


def test_index_is_per_notes_folder(open_db, config):
    config["dataset"] = {"path": "notes-a"}
    db = open_db()
    db.documents = make_docs()
    _ = db.db
    assert open_db().loaded_from_disk
    config["dataset"] = {"path": "notes-b"}
    assert not open_db().loaded_from_disk
