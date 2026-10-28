# conftest.py imports ragga before this module is collected, so the telemetry opt-outs ragga sets on import
# are already in place when these libraries read them.
import contextlib
import socket

import pytest
from huggingface_hub import constants
from unstructured.utils import _telemetry_opt_out

from tests.conftest import network_guard


def test_unstructured_telemetry_disabled():
    assert _telemetry_opt_out()


def test_huggingface_telemetry_disabled():
    assert constants.HF_HUB_DISABLE_TELEMETRY


def test_network_guard_catches_swallowed_attempts():
    with pytest.raises(AssertionError, match=r"DNS lookup 'telemetry\.invalid'"), network_guard():
        with contextlib.suppress(Exception):  # a library swallowing the error, like telemetry senders do
            socket.create_connection(("telemetry.invalid", 80), timeout=1)
