"""End-to-end smoke test with a real GGUF model. Opt-in: set RAGGA_TEST_MODEL=/path/to/model.gguf"""

import os

import pytest

from ragga import Config, Encoder, Generator, MarkdownDataset, VectorDatabase
from ragga.core.config import DEFAULT_CONFIG_PATH
from ragga.crafting.prompt import TinyLlamaChatPrompt

MODEL = os.environ.get("RAGGA_TEST_MODEL")

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL, reason="set RAGGA_TEST_MODEL to a GGUF model path"),
]


def test_answers_from_notes(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "pets.md").write_text("# Pets\n\nMy cat is called Biscuit and she is orange.\n")
    conf = Config(config_path=DEFAULT_CONFIG_PATH)
    conf.merge(
        {
            "encoder": {
                "model_path": "sentence-transformers/all-MiniLM-l6-v2",
                "index_device": None,
                "cache_dir": str(tmp_path / "cache"),
                "model_kwargs": {"device": os.environ.get("RAGGA_TEST_DEVICE", "cpu")},
            },
            "retriever": {"cache_dir": str(tmp_path / "cache"), "num_docs": 2},
            "dataset": {"path": str(notes), "show_progress": False},
            "generator": {
                "llm_path": MODEL,
                "max_tokens": 48,
                "temperature": 0.0,
                "gpu_layers": int(os.environ.get("RAGGA_TEST_GPU_LAYERS", "0")),
            },
        }
    )
    db = VectorDatabase(conf, Encoder(conf))
    db.documents = MarkdownDataset(conf).documents
    generator = Generator(conf, TinyLlamaChatPrompt(conf), db)

    streamed: list[str] = []
    generator.subscribe(streamed.append)
    answer = "".join(generator.get_answer_stream("What is my cat called?"))

    assert answer.strip()
    assert "".join(streamed).strip()
    assert "Biscuit" in (generator.last_context or "")

    # a follow-up sees the conversation so far
    "".join(generator.get_answer_stream("What colour is she?"))
    assert len(generator.history) == 2
    assert generator.history[0][1] in (generator.last_context or "")

    # continuing reuses the stored prompt + output, and extends the last answer
    before = generator.history[-1][1]
    "".join(generator.continue_stream())
    assert len(generator.history) == 2
    assert generator.history[-1][1].startswith(before)
