import ctypes
import os
import sys
from io import StringIO

import pytest

from ragga.core.io import store_stdout_stderr


def test_captures_python_output(capfd):
    with capfd.disabled(), store_stdout_stderr(close_buff=False) as (out, err):
        print("hello")  # noqa: T201
        print("oops", file=sys.stderr)  # noqa: T201
    assert out.getvalue() == "hello\n"
    assert err.getvalue() == "oops\n"


@pytest.mark.skipif(os.name == "nt", reason="libc printf capture is POSIX only")
def test_captures_c_output(capfd):
    libc = ctypes.CDLL(None)
    with capfd.disabled(), store_stdout_stderr(close_buff=False) as (out, _):
        libc.printf(b"from C\n")
    assert "from C" in out.getvalue()


def test_disabled_does_not_capture(capsys):
    out, err = StringIO(), StringIO()
    with store_stdout_stderr(out, err, disable=True):
        print("visible")  # noqa: T201
    assert out.getvalue() == ""
    assert capsys.readouterr().out == "visible\n"


def test_decode_normalises_line_endings_and_bad_utf8():
    # Windows C runtimes write \r\n in text mode; llama.cpp can emit a multi-byte character split across tokens
    assert store_stdout_stderr._decode(b"a\r\nb\n\xe2\x82") == "a\nb\n�"
