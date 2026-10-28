"""Minimal langchain LLM wrapper around an in-process llama.cpp model (llama-cpp-python)."""

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from llama_cpp import Llama, llama_model_default_params
from llama_cpp._internals import LlamaModel
from pydantic import ConfigDict, Field, PrivateAttr


class LlamaCpp(LLM):
    """llama.cpp model as a langchain LLM.

    `model_kwargs` are passed to `llama_cpp.Llama(...)` (e.g. flash_attn, n_threads, seed, use_mlock, verbose),
    `generate_kwargs` to each completion call (e.g. top_p, top_k, repeat_penalty).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_batch: int = 512
    temperature: float = 0.8
    max_tokens: int = 256
    streaming: bool = True
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    _client: Llama = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        trained = self.trained_context_length(self.model_path)
        if trained and self.n_ctx > trained:
            # past its training context a model produces gibberish (e.g. TinyLlama: 2048)
            logging.warning(f"context_length {self.n_ctx} is more than the model supports, using {trained}")
            self.n_ctx = trained
            if self.max_tokens > trained // 4:  # leave most of a small context for the notes
                logging.warning(f"max_tokens {self.max_tokens} is too long for this context, using {trained // 4}")
                self.max_tokens = trained // 4
        self._client = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            **self.model_kwargs,
        )

    @staticmethod
    def trained_context_length(model_path: str) -> int | None:
        """Context length the model was trained with, from the GGUF metadata (loads only the vocabulary)."""
        params = llama_model_default_params()
        params.vocab_only = True
        model = LlamaModel(path_model=model_path, params=params, verbose=False)  # no context: quick and quiet
        try:
            metadata = model.metadata()
        finally:
            model.close()
        value = metadata.get(f"{metadata.get('general.architecture')}.context_length")
        return int(value) if value else None

    @property
    def client(self) -> Llama:
        return self._client

    @property
    def _llm_type(self) -> str:
        return "llama.cpp"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_path": self.model_path, "n_ctx": self.n_ctx, "temperature": self.temperature}

    def _completion_kwargs(self, stop: list[str] | None) -> dict[str, Any]:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature, "stop": stop, **self.generate_kwargs}

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:  # so token callbacks fire for invoke() too
            return "".join(chunk.text for chunk in self._stream(prompt, stop, run_manager, **kwargs))
        result = self._client(prompt, **self._completion_kwargs(stop), **kwargs)
        return result["choices"][0]["text"]  # type: ignore[index]

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for part in self._client(prompt, stream=True, **self._completion_kwargs(stop), **kwargs):
            chunk = GenerationChunk(text=part["choices"][0]["text"])  # type: ignore[index]
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        return len(self._client.tokenize(text.encode("utf-8")))
