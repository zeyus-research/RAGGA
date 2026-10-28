import hashlib
import logging
import typing as t
from pathlib import Path
from types import MappingProxyType

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragga.core.config import Config, Configurable
from ragga.pipeline.encoder import Encoder
from ragga.pipeline.retrieval import HybridRetriever
from ragga.pipeline.vectorstore import NumpyVectorStore

# bump when the way passages are built changes, so existing indexes are rebuilt
PASSAGE_FORMAT_VERSION = 2


def passage_header(metadata: dict[str, t.Any]) -> str:
    """Note title and folder, put in front of every chunk so it carries the context of the note it came from."""
    if not metadata.get("title"):
        return ""
    folder = metadata.get("folder")
    return f"Note: {metadata['title']}" + (f" ({folder})" if folder else "") + "\n\n"


class VectorDatabase(Configurable):
    """Vector database: exact search over a numpy matrix, kept in sync with the documents by langchain's indexer"""

    _config_key = "retriever"
    _default_config = MappingProxyType(
        {
            "splitting": {
                "chunk_size": 1000,
                "chunk_overlap": 150,
            },
            "num_docs": 10,
            "search": "hybrid",  # "hybrid" (BM25 keywords + vectors) or "vector"
            "keyword_weight": 1.0,  # hybrid only: weight of the keyword ranking relative to the vector ranking
            "max_per_note": 4,  # hybrid only: max chunks from one note, null for no limit
            "cache_dir": ".cache",
        }
    )
    _db: NumpyVectorStore | None = None
    _retriever: BaseRetriever | None = None
    _passages: list[Document] | None = None
    _documents: list[Document]
    _db_is_stale: bool = True
    _encoder: Encoder
    _record_manager: SQLRecordManager
    _loaded_from_disk: bool = False

    def __init__(self, conf: Config, encoder: Encoder, namespace: str | None = None) -> None:
        super().__init__(conf)
        splitting = self.config[self._config_key]["splitting"]
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitting["chunk_size"],
            chunk_overlap=splitting["chunk_overlap"],
            add_start_index=True,
        )
        # one index per notes folder, embedding model and chunking: changing any of them builds a fresh index instead of
        # mixing them (the indexer only removes notes of the folder being synced)
        notes = self.config.config.get("dataset", {}).get("path")
        notes = str(Path(notes).expanduser().resolve()) if notes else ""
        fingerprint = hashlib.sha256(
            f"{notes}|{encoder.model_name}|{splitting['chunk_size']}|{splitting['chunk_overlap']}|"
            f"{PASSAGE_FORMAT_VERSION}".encode()
        ).hexdigest()[:12]
        cache_dir = Path(self.config[self.key]["cache_dir"])
        self._index_db_path = cache_dir / ".record_index.db"
        self._vector_db_path = cache_dir / ".vectors" / fingerprint
        self._encoder = encoder
        self._documents = []
        self._namespace = f"{namespace if namespace is not None else self._config_key}:{fingerprint}"
        self._record_manager = SQLRecordManager(
            namespace=self._namespace,
            db_url=f"sqlite:///{self._index_db_path}",
        )

        self._vector_db_path.mkdir(parents=True, exist_ok=True)
        self._create_db()

    @property
    def documents(self) -> list[Document]:
        return self._documents

    @documents.setter
    def documents(self, documents: list[Document]) -> None:
        """
        Adds documents to the database
        Args:
            documents (list): list of documents
        """
        self._documents = documents
        self._db_is_stale = True

    @property
    def passages(self) -> list[Document]:
        if self.documents is None or self.documents == []:
            return []
        if self._passages is None or self._db_is_stale:
            self._passages = self._text_splitter.split_documents(self.documents)
            for passage in self._passages:
                passage.metadata["header"] = header = passage_header(passage.metadata)
                passage.page_content = header + passage.page_content
        return self._passages

    @property
    def db(self) -> NumpyVectorStore:
        if self._db is None or self._db_is_stale:
            self._create_db()
        if self._db is None:
            msg = "Database failed to initialize."
            raise ValueError(msg)
        return self._db

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def _add_to_index(self, documents: list[Document], mode: t.Literal["incremental", "full"] = "incremental") -> None:
        """
        Adds documents to the index
        Args:
            documents (list): list of documents
        """
        if self._db is not None:
            index(
                documents,
                record_manager=self._record_manager,
                vector_store=self._db,
                cleanup=mode,
                source_id_key="source",
                key_encoder="sha256",
            )

    def _clear_index(self) -> None:
        """
        Clears the index
        """
        self._add_to_index([], mode="full")

    def _update_index(self) -> None:
        if self._db is None:
            msg = "Database failed to initialize."
            raise ValueError(msg)
        if self.passages is None or self.passages == []:
            return
        self._add_to_index(self.passages)
        self._db.save(self._vector_db_path)
        self._db_is_stale = False

    def _create_db(self) -> None:
        """
        Store passages in vector database in embedding format
        """
        self._record_manager.create_schema()
        if self._db is None:
            if NumpyVectorStore.exists(self._vector_db_path):
                logging.info("Loading vector database from disk")
                self._db = NumpyVectorStore.load(self._vector_db_path, self._encoder.embeddings)
                self._loaded_from_disk = True
                logging.debug(
                    "Vector database loaded from disk: "
                    "No documents will exist in this object, but they will be available in the database."
                )
            else:
                logging.info("Creating new vector database")
                self._db = NumpyVectorStore(self._encoder.embeddings)
                # The record manager remembers what was indexed. Without a stored index (first run, deleted cache,
                # or an index from an older RAGGA version) those records are stale and would make the indexer skip
                # every document, so start over. Embeddings are cached, so nothing gets re-encoded.
                if stale := self._record_manager.list_keys():
                    logging.info(f"Resetting {len(stale)} stale index records")
                    self._record_manager.delete_keys(stale)
        self._update_index()

    def merge_documents(self, documents: list[Document]) -> None:
        """
        Merges documents with existing documents
        Args:
            documents (list): list of documents
        """
        self._documents.extend(documents)
        self._db_is_stale = True
        self._update_index()

    def get_similar_docs(self, question: str, k: int | None = None) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to return
                     defaults to config value if None
        Returns:
            str: most similar document
        """
        k = self.config[self._config_key]["num_docs"] if k is None else k
        docs: list[Document] = self.db.similarity_search(question, k=k)
        strdocs: list[str] = [d.page_content for d in docs]

        return "\n".join(strdocs)

    def finish_indexing(self) -> None:
        """Index any pending documents, then let the encoder release the GPU (call before loading the LLM)."""
        _ = self.db
        self._encoder.finish_indexing()

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            conf = self.config[self._config_key]
            k = conf["num_docs"]
            if conf["search"] == "hybrid":
                self._retriever = HybridRetriever(
                    store=self.db,
                    k=k,
                    fetch_k=max(50, 4 * k),
                    keyword_weight=conf["keyword_weight"],
                    max_per_source=conf["max_per_note"],
                )
            elif conf["search"] == "vector":
                self._retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
            else:
                msg = f"Unknown retriever.search: {conf['search']!r}, use 'hybrid' or 'vector'"
                raise ValueError(msg)
        return self._retriever

    @property
    def loaded_from_disk(self) -> bool:
        return self._loaded_from_disk

    def close(self) -> None:
        """Close the record manager's SQLite connections (the vectors live in memory / on disk already)."""
        self._record_manager.engine.dispose()  # type: ignore[union-attr]

    def __enter__(self) -> t.Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
