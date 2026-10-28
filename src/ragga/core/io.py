"""This module contains the io class for storing stdout and stderr.

Thanks to:
- https://github.com/abetlen/llama-cpp-python/blob/fcdf337d84591c46c0a26b8660b2e537af1fd353/llama_cpp/_utils.py
- https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
- https://stackoverflow.com/questions/17942874/stdout-redirection-with-ctypes

MIT License

Copyright (c) 2024 zeyus (https://github.com/zeyus)
Copyright (c) 2023 Andrei Betlen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import ctypes
import os
import sys
import tempfile
from io import SEEK_SET, StringIO, TextIOWrapper

WIN: bool = os.name == "nt"
if not WIN:
    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
    c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")
else:
    if hasattr(sys, "gettotalrefcount"):
        libc = ctypes.CDLL("ucrtbase.dll")
    else:
        libc = ctypes.CDLL("msvcrt.dll")
    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    c_stdout = kernel32.GetStdHandle(-11)
    c_stderr = kernel32.GetStdHandle(-12)


class store_stdout_stderr(object):  # noqa: N801, UP004
    """Store stdout and stderr in buffers.

    This class is used to store stdout and stderr in buffers. It is used to
    capture the output of the C code as well.

    It is used as a context manager.

    Examples:

    >>> with store_stdout_stderr() as (outbuff, errbuff):
    ...     print("Hello World!")
    >>> outbuff.getvalue()
    'Hello World!\\n'
    >>> errbuff.getvalue()
    ''

    >>> from io import StringIO
    >>> outbuff = StringIO()
    >>> errbuff = StringIO()
    >>> with store_stdout_stderr(outbuff, errbuff):
    ...     print("Hello World!")
    >>> outbuff.getvalue()
    'Hello World!\\n'
    >>> errbuff.getvalue()
    ''
    """

    _sys = sys
    _os = os

    def __init__(
        self,
        outbuff: StringIO | None = None,
        errbuff: StringIO | None = None,
        disable: bool | None = None,
        close_buff: bool | None = None,
    ) -> None:
        """Initialize the class.

        Args:
            outbuff (StringIO, optional): The buffer to store stdout in.
                If None, a new StringIO is created. Defaults to None.
            errbuff (StringIO, optional): The buffer to store stderr in.
                If None, a new StringIO is created. Defaults to None.
            disable (bool, optional): Disable the redirection.
                If None, disable is set to False. Defaults to None.
            close_buff (bool, optional): Close the buffers on context __exit__.
                If None, close_buff defaults to True if disable is None, False otherwise.
                Defaults to None.
        """
        self.outbuff = outbuff if outbuff is not None else StringIO()
        self.errbuff = errbuff if errbuff is not None else StringIO()
        self.disable = disable if disable is not None else False
        self.close_buff = close_buff if close_buff is not None else disable is None

    def _flush_stdout_stderr(self) -> None:
        """Flush python and C stdout and stderr buffers."""
        self._sys.stdout.flush()
        self._sys.stderr.flush()
        if not WIN:
            libc.fflush(c_stdout)
            libc.fflush(c_stderr)
        else:
            libc.fflush(None)

    def __enter__(self) -> tuple[StringIO, StringIO]:
        if self.disable:
            return self.outbuff, self.errbuff
        # If stdout/err do not have a fileno, we cannot redirect them
        if not hasattr(self._sys.stdout, "fileno") or not hasattr(self._sys.stderr, "fileno"):
            self.disable = True
            return self.outbuff, self.errbuff

        # Save the original file descriptors
        self.old_stdout_fd = self._sys.stdout.fileno()
        self.old_stderr_fd = self._sys.stderr.fileno()

        # Duplicate the original file descriptors
        self.old_stdout_dup = self._os.dup(self.old_stdout_fd)
        self.old_stderr_dup = self._os.dup(self.old_stderr_fd)

        # Create temporary files to store stdout and stderr
        self.stdout_tfile = tempfile.TemporaryFile(mode="w+b")
        self.stderr_tfile = tempfile.TemporaryFile(mode="w+b")

        # Save the original stdout and stderr
        self.old_stdout = self._sys.stdout
        self.old_stderr = self._sys.stderr

        self._flush_stdout_stderr()

        # Redirect stdout and stderr file descriptors to the temporary files
        self._os.dup2(self.stdout_tfile.fileno(), self.old_stdout_fd)
        self._os.dup2(self.stderr_tfile.fileno(), self.old_stderr_fd)

        # Redirect stdout and stderr to the temporary files
        self._sys.stdout = TextIOWrapper(  # type: ignore
            self._os.fdopen(self.stdout_tfile.fileno(), "wb", closefd=False), encoding="utf-8", write_through=True
        )
        self._sys.stderr = TextIOWrapper(  # type: ignore
            self._os.fdopen(self.stderr_tfile.fileno(), "wb", closefd=False), encoding="utf-8", write_through=True
        )

        return self.outbuff, self.errbuff

    def __exit__(self, *_) -> None:
        if self.disable:
            return
        # Restore stdout and stderr
        self._flush_stdout_stderr()
        # Redirect stdout and stderr file descriptors to the saved file descriptors
        self._os.dup2(self.old_stdout_dup, self.old_stdout_fd)
        self._os.dup2(self.old_stderr_dup, self.old_stderr_fd)
        # return stdout and stderr to their original objects (TextIOWrapper)
        self._sys.stdout = self.old_stdout  # type: ignore
        self._sys.stderr = self.old_stderr  # type: ignore
        # Flush the temporary files
        self.stderr_tfile.flush()
        self.stdout_tfile.flush()
        # Seek to the beginning of the files
        self.stderr_tfile.seek(0, SEEK_SET)
        self.stdout_tfile.seek(0, SEEK_SET)
        # Write the temporary files to the buffers
        self.outbuff.write(self.stdout_tfile.read().decode())
        self.errbuff.write(self.stderr_tfile.read().decode())
        # Close the saved file descriptors
        self._os.close(self.old_stdout_dup)
        self._os.close(self.old_stderr_dup)
        # Close the temporary files
        self.stdout_tfile.close()
        self.stderr_tfile.close()
        # Close the buffers
        if self.close_buff:
            self.outbuff.close()
            self.errbuff.close()
