# SPDX-FileCopyrightText: 2023-present zeyus <zeyus@zeyus.com>
#
# SPDX-License-Identifier: MIT
import os
import sys

# This has to run before any dependency is imported. setdefault() keeps any explicit user choice.

# RAGGA is meant to keep everything local: opt out of library telemetry. `unstructured` otherwise
# reports every document it parses, and huggingface_hub reads DO_NOT_TRACK too.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# On macOS the torch wheel bundles its own libomp.dylib, and llama-cpp-python links Homebrew's libomp when it was
# built with OpenMP. LLVM's OpenMP runtime aborts the process when a second copy initialises ("OMP: Error #15").
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ragga.core.config import Config  # noqa: F401
from ragga.crafting.prompt import ChatPrompt, simple_prompt, simple_prompt_phi2  # noqa: F401
from ragga.dataset.dataset import MarkdownDataset  # noqa: F401
from ragga.pipeline.documents import VectorDatabase  # noqa: F401
from ragga.pipeline.encoder import Encoder  # noqa: F401
from ragga.pipeline.generation import Generator  # noqa: F401
from ragga.web.search import Search, WebSearchRetriever  # noqa: F401
