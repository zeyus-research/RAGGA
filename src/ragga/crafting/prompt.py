from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from langchain_core.prompts import (
    BasePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from llama_cpp.llama_types import ChatCompletionRequestMessage

from ragga.core.config import Config, Configurable

if TYPE_CHECKING:
    from llama_cpp import Llama

# temporary placeholder...we need local preprocessing
# to handle keywords etc (search? dates?)
simple_template = """
You are {user}'s AI assistant with access to the {user}'s personal notes and internet search.
Use the following context to help answer the {user}'s query at the end.

CONTEXT START

Today is {date}. Here are some excerpts from relevante notes:
{context}

CONTEXT END

Query: {question}
Answer: """

simple_prompt = PromptTemplate.from_template(simple_template)

simple_template_phi2 = """
Instruct: As {user}'s personal assistant, use the following to help answer their question: "{question}"

Relevant personal notes from the user:
{context}


{user}: {question}

Output:"""

simple_prompt_phi2 = PromptTemplate.from_template(simple_template_phi2)

# earlier (question, answer) turns of a conversation, oldest first
History = Sequence[tuple[str, str]]


def _escape(text: str) -> str:
    """Make literal text safe to put in a PromptTemplate (f-string format)."""
    return text.replace("{", "{{").replace("}", "}}")


class Prompt(ABC, Configurable):
    """Base prompt"""

    _config_key = "prompt"
    # whether get_prompt(history=...) puts earlier turns in the prompt, prompts without it are single-turn
    supports_history: bool = False

    @property
    def user_name(self):
        return self.config[self._config_key]["user_name"]

    @property
    def ai_name(self):
        return self.config[self._config_key]["AI_name"]

    def bind_llm(self, llm: "Llama") -> None:
        """Called by the generator once the model is loaded, for prompts that depend on the model."""

    @abstractmethod
    def get_prompt(self, **kwargs) -> BasePromptTemplate:
        """Get the prompt"""
        pass


class ChatPrompt(Prompt):
    """Chat prompt"""

    _default_config = MappingProxyType(
        {
            # "instruct_user": "Instruct",
            # "instructions":
            #     "You are \"<<AI_NAME>>\", {user}'s personal AI assistant. "
            #     "Use the following to help answer {user}'s question: \"{question}\"",
            # "pre_context": "Relevant personal notes from the user:",
            # "post_context": "",
            # "user_query": "{question}",
        }
    )

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        for k, v in self.config[self._config_key].items():
            if isinstance(v, str):
                self.config[self._config_key][k] = v.replace("<<AI_NAME>>", self.config[self._config_key]["AI_name"])

    def get_prompt(self, **_kwargs) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ChatMessagePromptTemplate.from_template(
                    role=self.config[self._config_key]["instruct_user"],
                    template="\n\n".join(
                        [
                            self.config[self._config_key]["instructions"],
                            self.config[self._config_key]["pre_context"],
                            "{context}",
                            self.config[self._config_key]["post_context"],
                        ]
                    ),
                ),
                ChatMessagePromptTemplate.from_template(
                    role=self.config[self._config_key]["AI_name"], template="How can I help?"
                ),
                ChatMessagePromptTemplate.from_template(
                    role=self.config[self._config_key]["user_name"] + " query:",
                    template=self.config[self._config_key]["user_query"],
                ),
            ]
        )


class Phi2QAPrompt(Prompt):
    """Prompt wiht Phi-2 specific instruct tokens

    Official format:
        Instruct: {prompt}
        Output:

    "where the model generates the text after "Output:".

    """

    _default_config = MappingProxyType({})

    def get_prompt(self, **kwargs) -> PromptTemplate:
        if kwargs.get("docs"):
            return PromptTemplate.from_template(
                "Instruct: Answer the question succinctly using the following document extracts:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]\n"
                "{question}\n"
                "Output: "
            )
        elif kwargs.get("web"):
            return PromptTemplate.from_template(
                "Instruct: Answer the question succinctly using the following search results:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]\n"
                "{question}\n"
                "Output: "
            )

        return PromptTemplate.from_template("Instruct: Answer the user succinctly.\n{context}{question}\nOutput: ")


class Phi2ChatPrompt(Prompt):
    """Prompt wiht Phi-2 specific instruct tokens

    Official format:
        Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
        Bob: Well, have you tried creating a study schedule and sticking to it?
        Alice: Yes, I have, but it doesn't seem to help much.
        Bob: Hmm, maybe you should try studying in a quiet environment, like the library.
        Alice: ...

    "where the model generates the text after the first Bob:".

    """

    _default_config = MappingProxyType({})

    def get_prompt(self, **kwargs) -> ChatPromptTemplate:
        if kwargs.get("docs"):
            return ChatPromptTemplate.from_messages(
                [
                    HumanMessagePromptTemplate.from_template(
                        "Answer the question succinctly using the following document extracts:\n"
                        "[DATA]\n"
                        "{context}\n"
                        "[/DATA]\n"
                        "{question}"
                    ),
                ]
            )
        elif kwargs.get("web"):
            return ChatPromptTemplate.from_messages(
                [
                    HumanMessagePromptTemplate.from_template(
                        "Answer the question succinctly using the following search results:\n"
                        "[DATA]\n"
                        "{context}\n"
                        "[/DATA]\n"
                        "{question}"
                    ),
                ]
            )

        return ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template("Answer the user succinctly.\n{context}{question}"),
            ]
        )


class TinyLlamaChatPrompt(Prompt):
    """Prompt wiht Tiny Llama specific instruct tokens

    Official format:
        <|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
    """

    _default_config = MappingProxyType({})

    supports_history = True

    def get_prompt(self, history: History = (), **kwargs) -> PromptTemplate:
        if kwargs.get("docs") or kwargs.get("web"):
            source = {
                (True, False): "document extracts",
                (False, True): "search results",
                (True, True): "search results and document extracts",
            }[bool(kwargs.get("docs")), bool(kwargs.get("web"))]
            system = (
                "You are a friendly and helpful AI assistant. "
                f"Answer the user succinctly using the following {source}:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]"
            )
            user = "{question}"
        else:
            system = "You are a friendly and helpful AI assistant. Answer the user succinctly."
            user = "{context}{question}"
        turns = "".join(f"<|user|>\n{_escape(q)}</s>\n<|assistant|>\n{_escape(a)}</s>\n" for q, a in history)
        return PromptTemplate.from_template(f"<|system|>\n{system}</s>\n{turns}<|user|>\n{user}</s>\n<|assistant|>")


class Llama2ChatPrompt(Prompt):
    """Prompt wiht Llama 2 specific instruct tokens

    Official format:
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message }} [/INST]

    Later turns follow as `{{ answer }} </s><s>[INST] {{ user_message }} [/INST]`.
    The BOS token (<s>) is added by llama.cpp, so it's not part of the template.
    """

    _default_config = MappingProxyType({})

    supports_history = True

    _template = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

    def get_prompt(self, history: History = (), **kwargs) -> PromptTemplate:
        """Get the prompt
        Args:
            history: Earlier (question, answer) turns, oldest first.
            docs: Whether to include the docs in the prompt.
            web: Whether to include the web search in the prompt.
        Returns:
            Constructed prompt. Accepts a "context" variable if docs or web is True.
        """
        if kwargs.get("docs") or kwargs.get("web"):
            source = {
                (True, False): "documents",
                (False, True): "search results",
                (True, True): "search results and documents",
            }[bool(kwargs.get("docs")), bool(kwargs.get("web"))]
            system = (
                "You are a friendly and helpful AI assistant. "
                f"Answer the user succinctly using the following {source}:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]"
            )
            user = "{question}"
        else:
            system = "You are a friendly and helpful AI assistant. Answer the user succinctly."
            user = "{context}{question}"
        if not history:
            return PromptTemplate.from_template(self._template.format(system=system, user=user))
        (first_q, first_a), *rest = history
        text = self._template.format(system=system, user=_escape(first_q))
        for q, a in rest:
            text += f" {_escape(first_a)} </s><s>[INST] {_escape(q)} [/INST]"
            first_a = a
        return PromptTemplate.from_template(f"{text} {_escape(first_a)} </s><s>[INST] {user} [/INST]")


class GGUFChatPrompt(Prompt):
    """Prompt using the chat template embedded in the GGUF model file (`tokenizer.chat_template`).

    Works for any chat model that ships a template (Mistral, Llama 3, Qwen, Gemma, TinyLlama...).
    The template is read when the generator binds the loaded model, so `get_prompt` can only be
    called after the `Generator` has been created.
    """

    _default_config = MappingProxyType({})

    supports_history = True

    _system = "You are a friendly and helpful AI assistant. Answer the user succinctly."
    # sentinels survive template rendering, then become prompt template variables
    _context_marker = "\x00RAGGA_CONTEXT\x00"
    _question_marker = "\x00RAGGA_QUESTION\x00"

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        self._formatter: Jinja2ChatFormatter | None = None
        self._bos_token = ""

    def bind_llm(self, llm: "Llama") -> None:
        template = llm.metadata.get("tokenizer.chat_template")
        if not template:
            msg = "The model has no embedded chat template (tokenizer.chat_template), use a model specific prompt."
            raise ValueError(msg)
        bos_id, eos_id = llm.token_bos(), llm.token_eos()
        self._bos_token = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
        eos_token = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
        self._formatter = Jinja2ChatFormatter(template=template, eos_token=eos_token, bos_token=self._bos_token)

    def _render(self, system: str, history: History, user: str) -> str:
        if self._formatter is None:
            msg = f"{self.__class__.__name__} needs a model, create the Generator before calling get_prompt."
            raise ValueError(msg)
        turns: list[ChatCompletionRequestMessage] = []
        for q, a in history:
            turns += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        turns.append({"role": "user", "content": user})
        try:
            text = self._formatter(messages=[{"role": "system", "content": system}, *turns]).prompt
        except ValueError:  # templates without a system role raise via raise_exception
            first = history[0][0] if history else user
            turns[0] = {"role": "user", "content": f"{system}\n\n{first}"}
            text = self._formatter(messages=turns).prompt
        # llama.cpp adds BOS itself when tokenizing
        if self._bos_token and text.startswith(self._bos_token):
            text = text[len(self._bos_token) :]
        return text

    def get_prompt(self, history: History = (), **kwargs) -> PromptTemplate:
        if kwargs.get("docs") and kwargs.get("web"):
            system = (
                f"{self._system} Answer from the web search results and the user's personal notes below. The search "
                "results come first, then extracts from the notes grouped by note, each starting with "
                "'## <note title> (<folder>)'. Only use the notes that are relevant, and say whether your answer comes "
                "from the web or from which notes.\n"
                f"[DATA]\n{self._context_marker}\n[/DATA]"
            )
            user = self._question_marker
        elif kwargs.get("docs"):
            system = (
                f"{self._system} Answer from the user's personal notes below. Extracts are grouped by note, each "
                "starting with '## <note title> (<folder>)', and '[...]' marks skipped text. Mention which notes you "
                "used, and say so if the notes don't cover the question rather than guessing.\n"
                f"[DATA]\n{self._context_marker}\n[/DATA]"
            )
            user = self._question_marker
        elif kwargs.get("web"):
            system = (
                f"{self._system} Use the following search results to answer:\n[DATA]\n{self._context_marker}\n[/DATA]"
            )
            user = self._question_marker
        else:
            system = self._system
            user = self._context_marker + self._question_marker
        text = _escape(self._render(system, history, user))
        text = text.replace(self._context_marker, "{context}").replace(self._question_marker, "{question}")
        return PromptTemplate.from_template(text)
