import gc
import logging
from types import MappingProxyType

import torch
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.stores import InMemoryByteStore
from langchain_huggingface import HuggingFaceEmbeddings

from ragga.core.config import Config, Configurable


class Encoder(Configurable):
    """Encoder to create word embeddings"""

    _config_key = "encoder"

    _store: LocalFileStore | InMemoryByteStore | None = None

    _core_embeddings: HuggingFaceEmbeddings

    _embeddings: CacheBackedEmbeddings | HuggingFaceEmbeddings

    _default_config = MappingProxyType(
        {
            "model_path": "sentence-transformers/all-MiniLM-l6-v2",
            "cache_embeddings": True,
            "persistent_cache": True,
            "cache_dir": "cache",
            # device used while indexing notes, e.g. "cuda" (also ROCm): much faster for large embedding models.
            # Once indexing is done the model moves to model_kwargs.device, leaving the GPU to the LLM.
            "index_device": None,
        }
    )

    _default_model_kwargs = MappingProxyType(
        {
            "device": "cuda",
        }
    )

    _default_encode_kwargs = MappingProxyType(
        {
            "normalize_embeddings": False,
        }
    )

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)

        self._merge_default_kwargs(dict(self._default_model_kwargs), "model_kwargs")
        self._merge_default_kwargs(dict(self._default_encode_kwargs), "encode_kwargs")

        self._query_device: str = self.config[self._config_key]["model_kwargs"]["device"]
        index_device = self.config[self._config_key].get("index_device") or self._query_device
        if index_device == "auto":
            index_device = self._detect_gpu() or self._query_device
        self._core_embeddings = HuggingFaceEmbeddings(
            model_name=self.config[self._config_key]["model_path"],
            model_kwargs={**self.config[self._config_key]["model_kwargs"], "device": index_device},
            encode_kwargs=self.config[self._config_key]["encode_kwargs"],
        )

        if self.config[self._config_key]["cache_embeddings"]:
            if self.config[self._config_key]["persistent_cache"]:
                self._store = LocalFileStore(self.config[self._config_key]["cache_dir"])
            else:
                self._store = InMemoryByteStore()
            self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                self._core_embeddings,
                self._store,
                namespace=self._core_embeddings.model_name,
                key_encoder="sha256",
            )
        else:
            self._embeddings = self._core_embeddings

    @staticmethod
    def _detect_gpu() -> str | None:
        if torch.cuda.is_available():  # NVIDIA, and AMD with a ROCm build of torch
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return None

    @property
    def model_name(self) -> str:
        return self._core_embeddings.model_name

    def finish_indexing(self) -> None:
        """Move the model from the index device to the query device, freeing GPU memory for the LLM."""
        model = self._core_embeddings._client
        if model.device.type == torch.device(self._query_device).type:
            return
        logging.info(f"Moving encoder to {self._query_device}")
        # half precision is only worth it on a GPU (and slow or unsupported on CPU)
        dtype = torch.float32 if self._query_device == "cpu" else None
        model.to(device=self._query_device, dtype=dtype)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @property
    def embeddings(self) -> CacheBackedEmbeddings | HuggingFaceEmbeddings:
        return self._embeddings
