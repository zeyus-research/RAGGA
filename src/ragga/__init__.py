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
os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")  # its "try our commercial offering" banner

# On macOS the torch wheel bundles its own libomp.dylib, and llama-cpp-python links Homebrew's libomp when it was
# built with OpenMP. LLVM's OpenMP runtime aborts the process when a second copy initialises ("OMP: Error #15").
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch before llama_cpp. On ROCm, torch bundles its own HIP runtime. If llama.cpp's HIP backend loads first it
# binds to the system ROCm, then torch loads a second runtime, and with two HIP runtimes in one process llama.cpp
# inference becomes 20-40x slower. Loaded in this order, llama.cpp reuses torch's runtime.
import torch  # noqa: F401

from ragga.core.config import Config  # noqa: F401
from ragga.crafting.prompt import ChatPrompt, GGUFChatPrompt, simple_prompt, simple_prompt_phi2  # noqa: F401
from ragga.dataset.dataset import MarkdownDataset  # noqa: F401
from ragga.pipeline.documents import VectorDatabase  # noqa: F401
from ragga.pipeline.encoder import Encoder  # noqa: F401
from ragga.pipeline.generation import Generator  # noqa: F401
from ragga.web.search import Search, WebSearchRetriever  # noqa: F401
