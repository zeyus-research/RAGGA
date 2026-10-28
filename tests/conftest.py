import contextlib
import socket
import threading

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragga import Config


class FakeEncoder:
    """Stands in for ragga.Encoder without downloading a sentence-transformers model."""

    model_name = "fake"

    def __init__(self, size: int = 32) -> None:
        self.embeddings = DeterministicFakeEmbedding(size=size)

    def finish_indexing(self) -> None:
        pass


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


@contextlib.contextmanager
def network_guard():
    """Fail if anything touches the network (DNS lookups included) inside the block, background threads too.

    Attempts are recorded rather than only raised, because libraries (e.g. telemetry senders) often swallow the
    exception.
    """
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):
        if self.family == socket.AF_UNIX:
            return original_connect(self, address, *args, **kwargs)
        attempts.append(f"connect {address!r}")
        msg = f"network access attempted during a unit test: {address!r}"
        raise RuntimeError(msg)

    def guarded_getaddrinfo(host, *args, **kwargs):
        attempts.append(f"DNS lookup {host!r}")
        msg = f"DNS lookup attempted during a unit test: {host!r}"
        raise socket.gaierror(msg)

    threads_before = set(threading.enumerate())
    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(timeout=2)
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.getaddrinfo = original_getaddrinfo
    assert not attempts, f"unexpected network access: {attempts}"


@pytest.fixture(autouse=True)
def no_network(request):
    """Unit tests must not touch the network (catches dependencies phoning home). Model tests are exempt."""
    if request.node.get_closest_marker("model"):
        yield
        return
    with network_guard():
        yield
