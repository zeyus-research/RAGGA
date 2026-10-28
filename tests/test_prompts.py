import pytest

from ragga import ChatPrompt
from ragga.crafting.prompt import GGUFChatPrompt, Llama2ChatPrompt, Phi2ChatPrompt, Phi2QAPrompt, TinyLlamaChatPrompt

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


MISTRAL_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]"
    "{% elif m['role'] == 'user' %}[INST]{{ m['content'] }}[/INST]"
    "{% endif %}{% endfor %}"
)
CHATML_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
NO_SYSTEM_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "<user>{{ m['content'] }}</user>{% endfor %}"
)


class FakeLlm:
    """Just enough of llama_cpp.Llama for GGUFChatPrompt.bind_llm"""

    def __init__(self, template):
        self.metadata = {"tokenizer.chat_template": template} if template else {}
        self._model = self
        self._tokens = {1: "<s>", 2: "</s>"}

    def token_bos(self):
        return 1

    def token_eos(self):
        return 2

    def token_get_text(self, token):
        return self._tokens[token]


def _gguf_prompt(config, template):
    prompt = GGUFChatPrompt(config)
    prompt.bind_llm(FakeLlm(template))
    return prompt


@pytest.mark.parametrize("template", [MISTRAL_TEMPLATE, CHATML_TEMPLATE, NO_SYSTEM_TEMPLATE])
@pytest.mark.parametrize("kwargs", [{"docs": True}, {"web": True}, {}])
def test_gguf_prompt_renders(config, template, kwargs):
    template = _gguf_prompt(config, template).get_prompt(**kwargs)
    assert set(template.input_variables) == {"context", "question"}
    text = template.format(context="CTX {braces}", question="Q?")
    assert "CTX {braces}" in text
    assert "Q?" in text
    assert not text.startswith("<s>")  # llama.cpp adds BOS itself


def test_gguf_prompt_mistral_format(config):
    text = _gguf_prompt(config, MISTRAL_TEMPLATE).get_prompt(docs=True).format(context="CTX", question="Q?")
    assert text.startswith("[SYSTEM_PROMPT]")
    assert "[DATA]\nCTX\n[/DATA][/SYSTEM_PROMPT]" in text
    assert text.endswith("[INST]Q?[/INST]")


def test_gguf_prompt_chatml_generation_prompt(config):
    text = _gguf_prompt(config, CHATML_TEMPLATE).get_prompt(docs=True).format(context="CTX", question="Q?")
    assert text.endswith("<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n")


def test_gguf_prompt_system_fallback(config):
    text = _gguf_prompt(config, NO_SYSTEM_TEMPLATE).get_prompt(docs=True).format(context="CTX", question="Q?")
    assert text.startswith("<user>")
    assert text.endswith("[/DATA]\n\nQ?</user>")


def test_gguf_prompt_template_with_literal_braces(config):
    template = "{% for m in messages %}{{ '{' }}{{ m['content'] }}{{ '}' }}{% endfor %}"
    text = _gguf_prompt(config, template).get_prompt().format(context="CTX", question="Q?")
    assert text == "{You are a friendly and helpful AI assistant. Answer the user succinctly.}{CTXQ?}"


def test_gguf_prompt_requires_model(config):
    with pytest.raises(ValueError, match="needs a model"):
        GGUFChatPrompt(config).get_prompt()
    with pytest.raises(ValueError, match="no embedded chat template"):
        GGUFChatPrompt(config).bind_llm(FakeLlm(None))
