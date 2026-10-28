import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragga import Config


class FakeEncoder:
    """Stands in for ragga.Encoder without downloading a sentence-transformers model."""

    def __init__(self, size: int = 32) -> None:
        self.embeddings = DeterministicFakeEmbedding(size=size)


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        config={
            "retriever": {
                "splitting": {"chunk_size": 64, "chunk_overlap": 8},
                "num_docs": 2,
                "cache_dir": str(tmp_path / "cache"),
            },
            "prompt": {
                "user_name": "User",
                "AI_name": "Assistant",
                "instruct_user": "Instruct",
                "instructions": "You are <<AI_NAME>>.",
                "pre_context": "Docs:",
                "post_context": "",
                "user_query": "{question}",
            },
        }
    )


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
