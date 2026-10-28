from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langchain_core.documents import Document
from tqdm import tqdm
from unstructured.partition.md import partition_md

from ragga.core.config import Config, Configurable


class MarkdownDataset(Configurable):
    """Markdown dataset"""

    _config_key = "dataset"
    _default_config = MappingProxyType(
        {
            "path": "data",
            "recursive": True,
            "show_progress": True,
            "use_multithreading": True,
        }
    )

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        self._documents: list[Document] | None = None
        default_loader_kwargs = {
            "mode": "single",
            "unstructured_kwargs": {
                "include_metadata": True,
                "languages": ["en"],
                "chunking_strategy": "by_title",
            },
        }
        self._merge_default_kwargs(default_loader_kwargs, "loader_kwargs")

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
            results = pool.map(self._load_file, files)
            if conf["show_progress"]:
                results = tqdm(results, total=len(files), desc="Loading notes", unit="file")
            self._documents = [doc for docs in results for doc in docs]
        if len(self._documents) == 0:
            msg = "No documents found in dataset path"
            raise ValueError(msg)

    @staticmethod
    def _find_files(path: Path, *, recursive: bool) -> list[Path]:
        """Markdown files under `path`, skipping hidden files and folders (e.g. .obsidian, .trash)."""
        candidates = path.rglob("*.md") if recursive else path.glob("*.md")
        return sorted(
            f for f in candidates if f.is_file() and not any(part.startswith(".") for part in f.relative_to(path).parts)
        )

    def _load_file(self, file: Path) -> list[Document]:
        """Parse one Markdown file with unstructured, either as one document ("single") or one per element."""
        loader_kwargs = self.config[self._config_key]["loader_kwargs"]
        elements = partition_md(filename=str(file), **loader_kwargs.get("unstructured_kwargs", {}))
        source = {"source": str(file)}
        if loader_kwargs.get("mode", "single") == "elements":
            docs = []
            for element in elements:
                metadata: dict[str, Any] = {**source, **element.metadata.to_dict(), "category": element.category}
                if element.id:
                    metadata["element_id"] = element.id
                docs.append(Document(page_content=str(element), metadata=metadata))
            return docs
        return [Document(page_content="\n\n".join(str(el) for el in elements), metadata=source)]

    def __repr__(self) -> str:
        return f"MarkdownDataset({self.config[self._config_key]['path']})"
