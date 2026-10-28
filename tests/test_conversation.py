from io import StringIO
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from ragga import Config
from ragga.core.config import Configurable
from ragga.pipeline.generation import Generator


class FakeChain:
    """Streams a canned answer the way the real chain does: store the prompt, then each output chunk."""

    def __init__(self, generator, answer, *, fail=False):
        self.generator = generator
        self.answer = answer
        self.fail = fail

    def stream(self, inputs):
        if self.fail:  # e.g. the web search failing before the LLM is called
            msg = "network down"
            raise ConnectionError(msg)
        self.generator._store_context(f"PROMPT {inputs['question']}")
        for token in self.answer.split(" "):
            self.generator._store_output(token + " ")
            yield token + " "


def make_generator(history_tokens=2048, *, web=False):
    """A Generator without a model: tokens are counted as words."""
    generator = Generator.__new__(Generator)
    Configurable.__init__(generator, Config(config={}))
    generator.config["generator"]["history_tokens"] = history_tokens
    generator._llm = SimpleNamespace(get_num_tokens=lambda text: len(text.split()))
    generator._model_stdout, generator._model_stderr, generator._autoflush = StringIO(), StringIO(), True
    generator._web_retriever = object() if web else None
    generator._history, generator._kept_history, generator.forgotten_turns = [], [], 0
    generator._web_search = generator._last_web = False
    generator._question, generator._last_context, generator._last_output, generator._docs = None, None, "", None
    generator._generations = 0
    generator._gen_chain = FakeChain(generator, "an answer")
    generator._continue_chain = FakeChain(generator, "and more")
    return generator


def ask(generator, question, **kwargs):
    return "".join(generator.get_answer_stream(question, **kwargs))


def test_turns_are_recorded_and_follow_ups_search_with_the_previous_question():
    generator = make_generator()
    ask(generator, "What is my cat called?")
    assert generator.history == [("What is my cat called?", "an answer ")]
    ask(generator, "What colour is she?")
    assert generator._kept_history == [("What is my cat called?", "an answer ")]
    assert generator._prepare_query("What colour is she?") == "What is my cat called?\nWhat colour is she?"
    assert len(generator.history) == 2


def test_web_queries_are_sent_as_is():
    generator = make_generator(web=True)
    ask(generator, "first")
    ask(generator, "weather in Aarhus", web=True)
    assert generator._prepare_query("weather in Aarhus") == "weather in Aarhus"


def test_web_search_must_be_enabled():
    with pytest.raises(ValueError, match="not enabled"):
        make_generator().get_answer_stream("q", web=True)


def test_oldest_turns_are_dropped_when_over_budget():
    generator = make_generator(history_tokens=8)  # each turn is "qN" + "an answer " = 3 tokens
    for i in range(4):
        ask(generator, f"q{i}")
    assert [q for q, _ in generator._kept_history] == ["q1", "q2"]
    assert generator.forgotten_turns == 1
    assert len(generator.history) == 4  # nothing is lost, it just isn't sent


def test_no_history_tokens_means_single_turn():
    generator = make_generator(history_tokens=0)
    ask(generator, "q0")
    ask(generator, "q1")
    assert generator._kept_history == []
    assert generator._prepare_query("q1") == "q1"


def test_continue_extends_the_last_answer():
    generator = make_generator()
    ask(generator, "q")
    "".join(generator.continue_stream())
    "".join(generator.continue_stream())
    assert generator.history == [("q", "an answer and more and more ")]


def test_continue_and_redo_need_a_previous_answer():
    generator = make_generator()
    with pytest.raises(ValueError, match="No answer"):
        generator.continue_stream()
    with pytest.raises(ValueError, match="No question"):
        generator.redo_stream()


def test_redo_replaces_the_last_answer_and_leaves_it_out_of_the_prompt():
    generator = make_generator()
    ask(generator, "q0")
    ask(generator, "q1")
    generator._gen_chain = FakeChain(generator, "better")
    "".join(generator.redo_stream())
    assert generator.history == [("q0", "an answer "), ("q1", "better ")]
    assert [q for q, _ in generator._kept_history] == ["q0"]


def test_failed_answers_are_not_recorded():
    generator = make_generator()
    ask(generator, "q0")
    generator._gen_chain = FakeChain(generator, "", fail=True)
    with pytest.raises(ConnectionError):
        ask(generator, "q1")
    with pytest.raises(ConnectionError):
        "".join(generator.redo_stream())
    assert generator.history == [("q0", "an answer ")]


def test_stopped_answers_keep_what_was_generated():
    generator = make_generator()
    stream = generator.get_answer_stream("q")
    next(stream)
    stream.close()  # what Ctrl-C while iterating amounts to
    assert generator.history == [("q", "an ")]


def test_reset_forgets_the_conversation():
    generator = make_generator()
    ask(generator, "q")
    generator.reset()
    assert generator.history == []
    assert generator.last_context is None
    assert generator.last_query is None
    ask(generator, "new topic")
    assert generator._prepare_query("new topic") == "new topic"


def test_web_answers_use_search_results_then_notes():
    generator = make_generator(web=True)
    generator._web_retriever = RunnableLambda(lambda q: [Document(page_content=f"web: {q}", metadata={"url": "u"})])
    generator._retriever = RunnableLambda(lambda q: [Document(page_content=f"note: {q}", metadata={"title": "t"})])
    docs = generator._search_web_and_notes("cats")
    assert [d.page_content for d in docs] == ["web: cats", "note: cats"]
