"""End-to-end smoke test with a real GGUF model. Opt-in: set RAGGA_TEST_MODEL=/path/to/model.gguf"""

import os

import pytest

from ragga import Config, Encoder, Generator, MarkdownDataset, VectorDatabase
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
    conf = Config(config_path="config.yaml")
    conf.merge(
        {
            "encoder": {
                "cache_dir": str(tmp_path / "cache"),
                "model_kwargs": {"device": os.environ.get("RAGGA_TEST_DEVICE", "cpu")},
            },
            "retriever": {"cache_dir": str(tmp_path / "cache"), "num_docs": 2},
            "dataset": {"path": str(notes), "show_progress": False},
            "generator": {
                "llm_path": MODEL,
                "max_tokens": 48,
                "temperature": 0.0,
                "similarity_threshold": 0.0,
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

    # "continue" reuses the stored prompt + output
    more = "".join(generator.get_answer_stream("continue"))
    assert isinstance(more, str)
