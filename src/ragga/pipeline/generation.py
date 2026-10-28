import logging
from collections.abc import Callable, Iterator
from io import StringIO
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.output_parsers import BaseTransformOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import Runnable

from ragga.core.config import Config, Configurable
from ragga.core.dispatch import PropertyWrapper
from ragga.core.io import store_stdout_stderr
from ragga.crafting.prompt import Prompt
from ragga.pipeline.documents import VectorDatabase
from ragga.pipeline.llm import LlamaCpp

# generator.model_kwargs that are per-completion options rather than llama_cpp.Llama() constructor arguments
COMPLETION_KWARGS = frozenset(
    {
        "echo",
        "frequency_penalty",
        "grammar",
        "logprobs",
        "min_p",
        "mirostat_eta",
        "mirostat_mode",
        "mirostat_tau",
        "presence_penalty",
        "repeat_penalty",
        "stop",
        "suffix",
        "tfs_z",
        "top_k",
        "top_p",
        "typical_p",
    }
)

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class ModelResponseHandler(BaseCallbackHandler):
    """Callback handler for model output."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model_response = PropertyWrapper[str](initial_value=None)

    @property
    def model_response(self) -> str | None:
        """Get the model response."""
        return self._model_response.__get__(None, None)

    @model_response.setter
    def model_response(self, value: str | None) -> None:
        """Set the model response."""
        self._model_response.__set__(self, value)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        logging.debug("LLM started...")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list["BaseMessage"]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        logging.info("Chat model started...")

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # text LLMs (llama.cpp) always emit str; list content only comes from multimodal chat models
        self.model_response = token if isinstance(token, str) else str(token)

    def on_llm_end(self, response: "LLMResult", **kwargs: Any) -> None:
        """Run when LLM ends running."""
        logging.debug("LLM ended...")

    def on_llm_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        logging.debug("Chain started...")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        logging.debug("Chain ended...")

    def on_chain_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when chain errors."""
        logging.error(f"Chain error: {error}")

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: "AgentAction", **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: "AgentFinish", **kwargs: Any) -> None:
        """Run on agent end."""


class Generator(Configurable):
    """Generator, aka LLM, to provide an answer based on some question and context"""

    _config_key = "generator"
    _default_config = MappingProxyType(
        {
            "llm_path": "llm",
            "hf_tokenizer": "",
            "context_length": 1024,
            "temperature": 0.9,
            "gpu_layers": 50,
            "max_tokens": 128,
            "history_tokens": 2048,
            "n_batch": 256,
            "llama": False,
            "autoflush": True,
        }
    )

    _retriever: BaseRetriever

    def __init__(
        self,
        conf: Config,
        prompt: Prompt,
        vectorstore: VectorDatabase,
        websearch: BaseRetriever | None = None,
    ) -> None:
        super().__init__(conf)

        default_kwargs = {
            "streaming": True,
            # llama.cpp logs (including per-token debug lines on GPU) go to stderr when True
            "verbose": False,
            "use_mlock": False,
            "n_threads": None,
            "seed": -1,
            # off by default in llama-cpp-python; 2-3x faster prompt processing and generation on GPU
            "flash_attn": True,
        }
        search_kwargs = {
            "k": self.config[vectorstore.key]["num_docs"],
        }
        self._model_stdout: StringIO = StringIO()
        self._model_stderr: StringIO = StringIO()
        self._web_retriever: BaseRetriever | None = websearch
        self._question: str | None = None
        # earlier (question, answer) turns, and those of them that fit in the prompt this turn
        self._history: list[tuple[str, str]] = []
        self._kept_history: list[tuple[str, str]] = []
        self.forgotten_turns: int = 0
        self._last_web: bool = False
        self._last_context: str | None = None
        self._last_output: str = ""
        self._generations: int = 0  # prompts sent to the LLM, to tell whether a stream got that far
        self._web_search: bool = False
        self._model_response_handler = ModelResponseHandler()
        self._merge_default_kwargs(default_kwargs, "model_kwargs")
        self._merge_default_kwargs(search_kwargs, "search_kwargs")
        self._autoflush: bool = True if self.config[self._config_key]["autoflush"] else False
        if not self._autoflush:
            logging.debug(
                "Autoflush disabled, please remember to flush stdout and stderr "
                f"buffers maunally with {self.__class__.__name__}.flush_stdout_stderr()"
            )
        # index the notes (on the encoder's index device) before the LLM claims the GPU
        vectorstore.finish_indexing()
        with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
            # load Llama from local file
            model_kwargs = dict(self.config[self._config_key]["model_kwargs"])
            streaming = model_kwargs.pop("streaming", True)
            generate_kwargs = {k: model_kwargs.pop(k) for k in COMPLETION_KWARGS & model_kwargs.keys()}
            self._llm: LlamaCpp = LlamaCpp(
                model_path=self.config[self._config_key]["llm_path"],
                n_ctx=self.config[self._config_key]["context_length"],
                temperature=self.config[self._config_key]["temperature"],
                n_gpu_layers=self.config[self._config_key]["gpu_layers"],
                max_tokens=self.config[self._config_key]["max_tokens"],
                callbacks=CallbackManager([self._model_response_handler]),
                n_batch=self.config[self._config_key]["n_batch"],
                streaming=streaming,
                model_kwargs=model_kwargs,
                generate_kwargs=generate_kwargs,
            )
        prompt.bind_llm(self._llm.client)
        if self.config[self._config_key]["history_tokens"] and not prompt.supports_history:
            logging.warning(f"{prompt.__class__.__name__} is single-turn: follow-up questions won't see earlier ones")
        self._llm_with_wrap: Runnable[LanguageModelInput, str | BaseMessage]
        # self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        #     self.config[self._config_key]["hf_tokenizer"]
        # )
        if self.config[self._config_key]["llama"]:
            logging.warning(
                "generator.llama is deprecated and ignored: use Llama2ChatPrompt, which emits the Llama 2 chat format"
            )
        self._llm_with_wrap = self._llm
        # create prompt template
        # from the LLM, not the config: both are capped to what the model supports
        self._max_prompt_length: int = (
            self._llm.n_ctx - self._llm.max_tokens - 100 - 1
        )  # placeholder, will need to calculate this
        self._prompt: Prompt = prompt
        self._template: BasePromptTemplate | None = None
        self._output_parser: BaseTransformOutputParser = StrOutputParser()
        self._vectorstore: VectorDatabase = vectorstore
        self._retriever: BaseRetriever
        self._docs: list[Document] | None = None
        self._retriever = self._vectorstore.retriever
        retrieval: Runnable = self._retriever
        if self._web_retriever is not None:
            retrieval = RunnableBranch((self._use_web_retriever, self._search_web_and_notes), self._retriever)
        self._inputs: Runnable = RunnableParallel(  # type: ignore
            {
                "question": itemgetter("question"),
                "context": itemgetter("question")
                | RunnableLambda(func=self._prepare_query)
                | RunnablePassthrough(func=self._set_template)
                | retrieval
                | self.condense_context,
            }  # type: ignore
        )

        self._gen_chain: Runnable = (
            self._inputs
            | RunnableLambda(func=self._get_template)
            | RunnablePassthrough(func=self._store_context)
            | self._llm_with_wrap
            | self._output_parser
            | RunnablePassthrough(func=self._store_output)
        )
        self._continue_chain: Runnable = (
            RunnableLambda(func=self._create_continuation)
            | RunnablePassthrough(func=self._store_context)
            | self._llm_with_wrap
            | self._output_parser
            | RunnablePassthrough(func=self._store_output)
        )

        self._chain: Runnable = self._gen_chain

    def _create_continuation(self, _x: Any) -> str:
        """Create a prompt based on the last context and output"""
        if self._last_context is None:
            msg = "No context to continue"
            raise ValueError(msg)
        prompt: str = self._last_context + self._last_output
        logging.debug("Calculating continuation length...")
        enc_prompt: list[int] = self._llm.client.tokenize(prompt.encode())
        if len(enc_prompt) > self._max_prompt_length:
            logging.debug("Continuation too long, truncating start...")
            bprompt: bytes = self._llm.client.detokenize(enc_prompt[-self._max_prompt_length :])
            prompt = bprompt.decode()
        logging.debug("Continuation length calculated")
        return prompt

    def _get_template(self, _x: Any) -> BasePromptTemplate:
        """Get the template"""
        if self._template is None:
            msg = "No template set."
            raise ValueError(msg)
        return self._template

    def _set_template(self, _x: str) -> None:
        """Get the template"""
        if self._use_web_retriever():
            self._template = self._prompt.get_prompt(web=True, docs=True, history=self._kept_history)
        else:
            self._template = self._prompt.get_prompt(docs=True, history=self._kept_history)

    def _trim_history(self, history: list[tuple[str, str]]) -> None:
        """Keep the newest turns that fit in generator.history_tokens, whole turns are dropped oldest first."""
        budget: int = self.config[self._config_key]["history_tokens"] or 0
        kept: list[tuple[str, str]] = []
        for q, a in reversed(history):
            budget -= self._llm.get_num_tokens(q) + self._llm.get_num_tokens(a)
            if budget < 0:
                break
            kept.append((q, a))
        self._kept_history = kept[::-1]
        self.forgotten_turns = len(history) - len(kept)

    def _can_continue(self) -> bool:
        """Check if the context can be continued"""
        if self._last_context is None:
            return False
        return True

    def _store_context(self, ctx: PromptValue | str) -> None:
        """
        Store the context
        Args:
            ctx (str): context
        """
        if isinstance(ctx, PromptValue):
            ctx = ctx.to_string()
        self._last_context = ctx
        self._last_output = ""
        self._generations += 1

    def _store_output(self, output: str) -> None:
        """
        Store the output
        Args:
            output (str): output
        """
        self._last_output += output

    def _use_web_retriever(self, _: Any = None) -> bool:
        """Get the retriever dropin"""
        if self._web_search and self._web_retriever is not None:
            logging.info("Using web retriever")
            return True
        logging.info("Using vector retriever")
        return False

    def _search_web_and_notes(self, query: str) -> list[Document]:
        """Web results first (they're what was asked for), then the notes fill whatever context is left."""
        if self._web_retriever is None:
            msg = "web search is not enabled"
            raise ValueError(msg)
        return self._web_retriever.invoke(query) + self._retriever.invoke(query)

    def _prepare_query(self, x: str) -> str:
        """
        Prepare the query for the retriever. Notes are searched with the full question (BM25 discounts common words
        and embeddings work best on whole sentences), and a follow-up ("what about the second one?") together with
        the previous question, so it finds the notes the conversation is about. Web searches use the query as is.
        """
        if not self._web_search and self._kept_history:
            x = f"{self._kept_history[-1][0]}\n{x}"
        logging.info(f"Searching {'the web' if self._web_search else 'notes'} for: {x}")
        return x

    def flush_stdout_stderr(self) -> tuple[str, str]:
        """Flush stdout and stderr"""
        stdout = self._model_stdout.getvalue()
        stderr = self._model_stderr.getvalue()
        self._model_stdout.truncate(0)
        self._model_stderr.truncate(0)
        return stdout, stderr

    def condense_context(self, ctx: list[Document]) -> str:
        """
        Condense the context into a single string
        Args:
            ctx (list[Document]): list of documents
        Returns:
            str: condensed context
        """
        if self._template is None:
            msg = "No template set."
            raise ValueError(msg)
        self._docs = ctx
        logging.debug("Calculating condensed context...")
        template: str = self._template.format(question=self._question, context="")
        template_tokens: int = self._llm.get_num_tokens(template)
        available_tokens: int = self._max_prompt_length - template_tokens
        # add whole notes, most relevant first, until the context is full
        blocks: list[str] = []
        for block in self._group_by_note(ctx):
            block_tokens = self._llm.get_num_tokens(block) + 2
            if block_tokens > available_tokens:
                if not blocks:  # the best match alone is too long: keep its start
                    logging.debug("Context too long, truncating...")
                    tokens = self._llm.client.tokenize(block.encode(), add_bos=False)[: max(available_tokens, 0)]
                    blocks.append(self._llm.client.detokenize(tokens).decode(errors="ignore"))
                break
            blocks.append(block)
            available_tokens -= block_tokens
        logging.debug("Condensed context calculated")
        return "\n\n".join(blocks)

    @staticmethod
    def _group_by_note(docs: list[Document]) -> list[str]:
        """One text block per note, in order of each note's best match, with its chunks in document order."""
        notes: dict[str, list[Document]] = {}
        for i, doc in enumerate(docs):
            notes.setdefault(doc.metadata.get("source") or f"#{i}", []).append(doc)
        blocks = []
        for chunks in notes.values():
            chunks.sort(key=lambda d: d.metadata.get("start_index", 0))
            header = chunks[0].metadata.get("header", "")
            body, end = "", None
            for d in chunks:
                text = d.page_content.removeprefix(header)
                start = d.metadata.get("start_index")
                if end is not None and start is not None and start < end:  # overlaps the previous chunk
                    body += text[end - start :]
                else:
                    body += ("\n[...]\n" if body else "") + text
                if start is not None:
                    end = max(end or 0, start + len(text))
            title = header.strip().removeprefix("Note: ")
            blocks.append(f"## {title}\n{body.strip()}" if title else body.strip())
        return blocks

    @property
    def last_docs(self) -> list[Document] | None:
        """Get the last docs"""
        return self._docs

    @property
    def last_context(self) -> str | None:
        """Get the last context"""
        return self._last_context

    @property
    def last_query(self) -> str | None:
        """Get the last question"""
        return self._history[-1][0] if self._history else None

    @property
    def history(self) -> list[tuple[str, str]]:
        """The conversation so far: (question, answer) turns, oldest first"""
        return list(self._history)

    def reset(self) -> None:
        """Forget the conversation, the next question starts a new one."""
        self._history.clear()
        self._kept_history = []
        self.forgotten_turns = 0
        self._last_context = None
        self._last_output = ""
        self._docs = None

    def _ask(self, question: str, web: bool, *, redo: bool = False) -> None:
        if web and self._web_retriever is None:
            msg = "web search is not enabled"
            raise ValueError(msg)
        self._question = question
        self._web_search = self._last_web = web
        self._chain = self._gen_chain
        self._trim_history(self._history[:-1] if redo else self._history)

    def get_answer(self, question: str, *, web: bool = False) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            question (str): user's question
            web (bool): answer from a web search instead of the notes
        Returns:
            str: llm answer
        """
        self._ask(question, web)
        with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
            answer = self._chain.invoke({"question": self._question})
        self._history.append((question, self._last_output))
        return answer

    def get_answer_stream(self, question: str, *, web: bool = False) -> Iterator[str]:
        """
        Get the answer from llm based on context and user's question
        Args:
            question (str): user's question
            web (bool): answer from a web search instead of the notes
        Returns:
            Iterator[str]: llm answer
        """
        self._ask(question, web)
        return self._stream("new")

    def continue_stream(self) -> Iterator[str]:
        """Continue the last answer (e.g. when it was cut off at max_tokens)."""
        if not self._can_continue() or not self._history:
            msg = "No answer to continue"
            raise ValueError(msg)
        self._chain = self._continue_chain
        return self._stream("continue")

    def redo_stream(self) -> Iterator[str]:
        """Answer the last question again, its previous answer is replaced."""
        if not self._history:
            msg = "No question to answer again"
            raise ValueError(msg)
        self._ask(self._history[-1][0], self._last_web, redo=True)
        return self._stream("redo")

    def _stream(self, turn: str) -> Iterator[str]:
        """Stream the chain, and record the turn ("new", "continue" or "redo") once it's generating, even if stopped."""
        generations = self._generations
        try:
            # yield inside the context manager: the stream is lazy, so generation happens while iterating
            with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
                yield from self._chain.stream({"question": self._question})
        finally:
            if self._generations != generations:  # not if it failed before generating (e.g. the web search)
                if turn == "new":
                    self._history.append((self._question or "", self._last_output))
                else:
                    q, a = self._history.pop()
                    self._history.append((q, a + self._last_output if turn == "continue" else self._last_output))

    def response_handler(self) -> ModelResponseHandler:
        """Get the response handler"""
        return self._model_response_handler

    def subscribe(self, subscriber: Callable[[str], None]):
        """Subscribe to updates of the associated property."""
        self._model_response_handler._model_response.subscribe(subscriber)
