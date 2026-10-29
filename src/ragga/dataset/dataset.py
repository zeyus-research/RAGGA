from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langchain_core.documents import Document
from tqdm import tqdm
from unstructured.partition.md import partition_md

from ragga.core.config import Config, Configurable


class NoteDataset(Configurable, ABC):
    """A folder of notes, subclasses turn one file into documents (see `_load_file`).

    `extensions` are the file types the subclass reads by default, `dataset.extensions` in the config overrides them.
    """

    _config_key = "dataset"
    _default_config = MappingProxyType(
        {
            "path": "data",
            "recursive": True,
            "show_progress": True,
            "use_multithreading": True,
        }
    )
    extensions: tuple[str, ...] = ()

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        self._documents: list[Document] | None = None
        configured = self.config[self._config_key].get("extensions") or self.extensions
        self._extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in configured}
        if not self._extensions:
            msg = f"{self.__class__.__name__} has no file extensions, set dataset.extensions"
            raise ValueError(msg)

    @property
    def documents(self) -> list[Document]:
        if self._documents is None:
            self._load_documents()
        return self._documents  # type: ignore

    def _load_documents(self) -> None:
        path = Path(self.config[self._config_key]["path"])
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise ValueError(msg)
        if not path.is_dir():
            msg = f"Path is not a directory: {path}"
            raise ValueError(msg)

        conf = self.config[self._config_key]
        files = self._find_files(path, recursive=conf["recursive"])
        workers = 4 if conf["use_multithreading"] else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda f: self._load_file(f, path), files)
            if conf["show_progress"]:
                results = tqdm(results, total=len(files), desc="Loading notes", unit="file")
            self._documents = [doc for docs in results for doc in docs]
        if len(self._documents) == 0:
            msg = "No documents found in dataset path"
            raise ValueError(msg)

    def _find_files(self, path: Path, *, recursive: bool) -> list[Path]:
        """Note files under `path`, skipping hidden files and folders (e.g. .obsidian, .trash)."""
        candidates = path.rglob("*") if recursive else path.glob("*")
        return sorted(
            f
            for f in candidates
            if f.suffix.lower() in self._extensions
            and f.is_file()
            and not any(part.startswith(".") for part in f.relative_to(path).parts)
        )

    @staticmethod
    def _source_metadata(file: Path, root: Path) -> dict[str, Any]:
        # the note title and folder give chunks their context, e.g. "RiseTogether" in "10 PhD/RiseTogether"
        folder = file.parent.relative_to(root).as_posix()
        return {"source": str(file), "title": file.stem, "folder": "" if folder == "." else folder}

    @abstractmethod
    def _load_file(self, file: Path, root: Path) -> list[Document]:
        """Documents for one note file, with at least `_source_metadata(file, root)` in their metadata."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config[self._config_key]['path']})"


class MarkdownDataset(NoteDataset):
    """Markdown dataset"""

    extensions = (".md", ".markdown")

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        default_loader_kwargs = {
            "mode": "single",
            "unstructured_kwargs": {
                "include_metadata": True,
                "languages": ["en"],
                "chunking_strategy": "by_title",
            },
        }
        self._merge_default_kwargs(default_loader_kwargs, "loader_kwargs")

    def _load_file(self, file: Path, root: Path) -> list[Document]:
        """Parse one Markdown file with unstructured, either as one document ("single") or one per element."""
        loader_kwargs = self.config[self._config_key]["loader_kwargs"]
        elements = partition_md(filename=str(file), **loader_kwargs.get("unstructured_kwargs", {}))
        source = self._source_metadata(file, root)
        if loader_kwargs.get("mode", "single") == "elements":
            docs = []
            for element in elements:
                metadata: dict[str, Any] = {**source, **element.metadata.to_dict(), "category": element.category}
                if element.id:
                    metadata["element_id"] = element.id
                docs.append(Document(page_content=str(element), metadata=metadata))
            return docs
        return [Document(page_content="\n\n".join(str(el) for el in elements), metadata=source)]


class TextDataset(NoteDataset):
    """Any text notes, read as-is without parsing, e.g. `extensions: [".ml", ".mli"]` for OCaml notes."""

    extensions = (".txt",)

    def _load_file(self, file: Path, root: Path) -> list[Document]:
        text = file.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata=self._source_metadata(file, root))]


# `dataset.format` in the config, register your own NoteDataset subclass here to make it selectable
DATASET_FORMATS: dict[str, type[NoteDataset]] = {"markdown": MarkdownDataset, "text": TextDataset}


def load_dataset(conf: Config) -> NoteDataset:
    """The dataset for the format configured in `dataset.format` (default markdown)."""
    fmt = conf["dataset"].get("format", "markdown") if "dataset" in conf else "markdown"
    if fmt not in DATASET_FORMATS:
        msg = f'unknown dataset.format "{fmt}", use one of: {", ".join(DATASET_FORMATS)}'
        raise ValueError(msg)
    return DATASET_FORMATS[fmt](conf)
