from abc import ABC, abstractmethod
from types import MappingProxyType

from langchain_core.prompts import (
    BasePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

from ragga.core.config import Config, Configurable

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


class Prompt(ABC, Configurable):
    """Base prompt"""

    _config_key = "prompt"

    @property
    def user_name(self):
        return self.config[self._config_key]["user_name"]

    @property
    def ai_name(self):
        return self.config[self._config_key]["AI_name"]

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

    def get_prompt(self, **kwargs) -> PromptTemplate:
        if kwargs.get("docs"):
            return PromptTemplate.from_template(
                "<|system|>\n"
                "You are a friendly and helpful AI assistant. "
                "Answer the user succinctly using the following document extracts:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]</s>\n"
                "<|user|>\n"
                "{question}</s>\n"
                "<|assistant|>"
            )
        elif kwargs.get("web"):
            return PromptTemplate.from_template(
                "<|system|>\n"
                "You are a friendly and helpful AI assistant. "
                "Answer the user succinctly using the following search results:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]</s>\n"
                "<|user|>\n"
                "{question}</s>\n"
                "<|assistant|>"
            )

        return PromptTemplate.from_template(
            "<|system|>\n"
            "You are a friendly and helpful AI assistant. Answer the user succinctly.</s>\n"
            "<|user|>\n"
            "{context}{question}</s>\n"
            "<|assistant|>"
        )


class Llama2ChatPrompt(Prompt):
    """Prompt wiht Llama 2 specific instruct tokens

    Official format:
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message }} [/INST]

    The BOS token (<s>) is added by llama.cpp, so it's not part of the template.
    """

    _default_config = MappingProxyType({})

    _template = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

    def get_prompt(self, **kwargs) -> PromptTemplate:
        """Get the prompt
        Args:
            docs: Whether to include the docs in the prompt.
            web: Whether to include the web search in the prompt.
        Returns:
            Constructed prompt. Accepts a "context" variable if docs or web is True.
        """
        if kwargs.get("docs") or kwargs.get("web"):
            source = "documents" if kwargs.get("docs") else "search results"
            system = (
                "You are a friendly and helpful AI assistant. "
                f"Answer the user succinctly using the following {source}:\n"
                "[DATA]\n"
                "{context}\n"
                "[/DATA]"
            )
            return PromptTemplate.from_template(self._template.format(system=system, user="{question}"))

        return PromptTemplate.from_template(
            self._template.format(
                system="You are a friendly and helpful AI assistant. Answer the user succinctly.",
                user="{context}{question}",
            )
        )
