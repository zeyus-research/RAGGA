import pytest

from ragga import ChatPrompt
from ragga.crafting.prompt import Llama2ChatPrompt, Phi2ChatPrompt, Phi2QAPrompt, TinyLlamaChatPrompt

PROMPTS = [Phi2QAPrompt, Phi2ChatPrompt, TinyLlamaChatPrompt, Llama2ChatPrompt]


@pytest.mark.parametrize("prompt_cls", PROMPTS)
@pytest.mark.parametrize("kwargs", [{"docs": True}, {"web": True}, {}])
def test_prompt_renders(config, prompt_cls, kwargs):
    template = prompt_cls(config).get_prompt(**kwargs)
    assert set(template.input_variables) == {"context", "question"}
    text = template.format(context="CTX", question="Q?")
    assert "CTX" in text
    assert "Q?" in text


def test_tinyllama_format(config):
    text = TinyLlamaChatPrompt(config).get_prompt(docs=True).format(context="CTX", question="Q?")
    assert text.startswith("<|system|>")
    assert text.endswith("<|user|>\nQ?</s>\n<|assistant|>")


def test_llama2_format(config):
    text = Llama2ChatPrompt(config).get_prompt(docs=True).format(context="CTX", question="Q?")
    assert text.startswith("[INST] <<SYS>>\n")
    assert text.endswith("<</SYS>>\n\nQ? [/INST]")


def test_chat_prompt_substitutes_ai_name(config):
    prompt = ChatPrompt(config)
    assert prompt.user_name == "User"
    assert prompt.ai_name == "Assistant"
    text = prompt.get_prompt().format(context="CTX", question="Q?")
    assert "You are Assistant." in text
    assert "CTX" in text
