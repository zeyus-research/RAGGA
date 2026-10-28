# RAGGA: Retrieval Augmented Generation General Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/ragga.svg)](https://pypi.org/project/ragga)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ragga.svg)](https://pypi.org/project/ragga)

-----

**Table of Contents**

- [RAGGA: Retrieval Augmented Generation General Assistant](#ragga-retrieval-augmented-generation-general-assistant)
  - [RAGGA](#ragga)
  - [Example Usage](#example-usage)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from a release](#install-from-a-release)
    - [Development setup (uv)](#development-setup-uv)
    - [GPU acceleration for llama.cpp](#gpu-acceleration-for-llamacpp)
  - [Running the tests](#running-the-tests)
  - [Releasing](#releasing)
  - [Model Selection](#model-selection)
  - [License](#license)

## RAGGA

What is this? Load quantized LLMs and run them on your local devices. Interact with your own notes (currently markdown notes are supported by default) and ask questions about what you have written and get relevant answers from what you have written! Recommended model is TinyLlama-chat https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/README.md based on its speed of token generation and it's low system requirements. Try the 8-bit quantized model to start with and if that's still too beefy you can always try 5, 4, etc.

Basically, RAGGA is an local LLM based AI assistant that cares about your privacy, and does not require running things via untrusted third party APIs. The one exception is the web search feature, which can be completely disabled, so nothing ever leaves your system. Telemetry in
dependencies (`unstructured`, Hugging Face Hub) is switched off when RAGGA is imported (`DO_NOT_TRACK=1`), and the test
suite fails if anything tries to reach the network.

## Example Usage

Download one of the TinyLlama-chat quantized model (e.g. [8-bit quantized TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf)) and copy it to the models directory.

Install RAGGA (see [Installation](#installation)).

Edit the [`config.yaml`](config.yaml) file to point to the model you downloaded in the `generator:` `llm_path:`, and update the `dataset:` `path:` key to point to your directory containing markdown files.

Run the [`tinyllama_example.py`](tinyllama_example.py) script to try it out.

## Installation

RAGGA supports Python 3.12+ on Linux, macOS and Windows. Pick **one** extra for the PyTorch build
(used by the sentence-transformers embedding model):

| extra  | PyTorch build                              | platforms                          |
|--------|--------------------------------------------|------------------------------------|
| `cpu`  | CPU only                                   | Linux, Windows, macOS (MPS on Apple Silicon) |
| `gpu`  | NVIDIA CUDA 13.0                           | Linux, Windows                     |
| `rocm` | AMD ROCm 7.2                               | Linux                              |

### Prerequisites

`llama-cpp-python` is compiled from source on install, so you need a C/C++ compiler and CMake
(Visual Studio 2022 Build Tools with "Desktop development with C++" on Windows, `build-essential`/`base-devel` on
Linux, Xcode command line tools on macOS).

### Install from a release

Releases are published on [GitHub](https://github.com/zeyus/RAGGA/releases) (not PyPI). With [uv](https://docs.astral.sh/uv/):

```console
uv pip install "ragga[cpu] @ https://github.com/zeyus/RAGGA/releases/download/v0.1.0/ragga-0.1.0-py3-none-any.whl"
```

or with pip (the torch index is needed for the CPU/CUDA/ROCm builds on Linux/Windows):

```console
pip install --extra-index-url https://download.pytorch.org/whl/cpu "ragga[cpu] @ https://github.com/zeyus/RAGGA/releases/download/v0.1.0/ragga-0.1.0-py3-none-any.whl"
```

Every release has a [build provenance attestation](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations), verify a downloaded wheel with:

```console
gh attestation verify ragga-0.1.0-py3-none-any.whl --repo zeyus/RAGGA
```

### Development setup (uv)

```console
git clone https://github.com/zeyus/RAGGA.git && cd RAGGA
uv sync --extra cpu        # or --extra gpu / --extra rocm
uv run --extra cpu python tinyllama_example.py
```

Always pass the same `--extra` to `uv run` (or use `uv run --no-sync`), otherwise uv re-syncs to the default PyPI
torch build. Dependencies are locked with hashes in `uv.lock`, and uv is configured with a 7 day cooldown
(`exclude-newer`), so brand-new (possibly compromised) releases are never picked up by `uv lock --upgrade`.

### GPU acceleration for llama.cpp

The PyTorch extra only accelerates the embedding model. To offload the LLM to a GPU, rebuild `llama-cpp-python` with
the matching backend and set `generator.gpu_layers` in `config.yaml` (`-1` = all layers):

```console
# NVIDIA (CUDA toolkit installed)
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra gpu --reinstall-package llama-cpp-python

# AMD via HIP/ROCm (see "AMD: Vulkan or ROCm?" below for the system packages).
# Set GPU_TARGETS to your card (`rocminfo | grep gfx`), e.g. gfx1100 for RX 7900, gfx1102 for RX 7600, gfx1201 for RX 9070
CMAKE_ARGS="-DGGML_HIP=on -DGPU_TARGETS=gfx1102 -DGGML_HIP_ROCWMMA_FATTN=on" \
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  uv sync --extra rocm --reinstall-package llama-cpp-python

# Any vendor via Vulkan (Vulkan SDK / glslc installed), also works on Windows
CMAKE_ARGS="-DGGML_VULKAN=on" uv sync --extra cpu --reinstall-package llama-cpp-python

# Apple Silicon: Metal is enabled by default, nothing to do
```

uv caches the built wheel without looking at `CMAKE_ARGS`, so run `uv cache clean llama-cpp-python` before switching
backends, otherwise you silently get the previous (e.g. CPU-only) build back. Check with
`uv run --no-sync python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`.

#### AMD: Vulkan or ROCm?

Both work, and which one is faster depends on the card and model size, so it's worth benchmarking your own models
(see the table below).

- **Vulkan** only needs the Mesa RADV driver (`vulkan-radeon`) and `glslc`/Vulkan headers to build. Simplest setup.
- **ROCm/HIP** needs the HIP runtime plus hipBLAS at build *and* run time. On Arch/Manjaro the minimal set is
  `sudo pacman -S rocm-hip-runtime rocm-cmake hipblas rocwmma` (~2 GB installed, mostly rocBLAS/rocSOLVER kernels).
  You don't need the `rocm-hip-sdk`/`rocm-hip-libraries` meta packages or `hipblaslt`. `rocwmma` is optional, it
  enables the faster flash attention kernels (`-DGGML_HIP_ROCWMMA_FATTN=on`) on RDNA3/RDNA4.

The `rocm` extra's PyTorch wheels bundle their own ROCm runtime, so the embedding model runs on the GPU either way.

Measured on an RX 7600 XT (16 GB, RDNA3/gfx1102), i7-7700, Mesa 26.2 RADV, ROCm 7.2, llama-cpp-python 0.3.36,
flash attention on. ~1900 token prompt then 100 generated tokens (same shape as `koboldcpp --benchmark` at 2048
context), best of 2 runs after a warm-up, in tokens/second (prompt processing / generation). koboldcpp 1.120 is
included as a reference point.

| backend                     | TinyLlama 1.1B Q4_K_M | Qwen3.5 9B Q8_0 |
|-----------------------------|----------------------:|----------------:|
| RAGGA, CPU only             |           134 / 21.7  |               – |
| RAGGA, Vulkan               |         1,146 / 59.4  |      932 / 12.1 |
| **RAGGA, HIP/ROCm**         |     **7,844 / 164.2** | **1,369 / 23.0** |
| koboldcpp, Vulkan           |         7,492 / 140.9 |      637 / 12.3 |
| koboldcpp, ROCm             |         7,919 / 143.7 |    1,086 / 24.0 |

On this RDNA3 card ROCm is roughly twice as fast for generation. Results on RDNA4 (RX 9000 series) are different,
where Vulkan is often as fast or faster for small and mid-sized models (see e.g.
[this comparison](https://runaihome.com/blog/rdna4-vulkan-vs-rocm-local-llm-benchmark-2026/)), so measure your own
setup with [`scripts/05_benchmark_backends.py`](scripts/05_benchmark_backends.py):

```console
UV_PROJECT_ENVIRONMENT=.venv-hip uv run --no-sync python scripts/05_benchmark_backends.py models/your-model.gguf
```

Tip: build each backend into its own environment (`UV_PROJECT_ENVIRONMENT=.venv-hip uv sync ...`) to compare them
side by side.

On Windows PowerShell use `$env:CMAKE_ARGS = "-DGGML_CUDA=on"` before running `uv sync`.

## Running the tests

```console
uv run --extra cpu pytest                     # unit tests, no model download needed
uv run --extra cpu ruff check && uv run --extra cpu mypy src/ragga
RAGGA_TEST_MODEL=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf uv run --extra cpu pytest tests/test_generator_smoke.py
```

CI runs the test suite on Linux, macOS and Windows (Python 3.12 and 3.14) plus a `pip-audit` of the lock file on every
push and pull request. Dependabot opens monthly grouped update PRs (also with a 7 day cooldown).

## Releasing

1. `uv version --bump minor` (or patch/major), commit `pyproject.toml` + `uv.lock`.
2. `git tag v0.1.0 && git push --tags`

The release workflow re-runs CI, checks the tag matches the version, builds the wheel/sdist, attests them and attaches
them to a GitHub release.

## Model Selection

Subjective evaluation of 3 different models was performed by generating responses to general and dataset-specific questions for 4 publicly available Obsidian notes repositories. The results for response quality were ranked from 0-10 for each question, responses were shown anonymously so no indication of which model generated the responses was visible.

![reports/scores_by_model.png](reports/scores_by_model.png)

The phi-2 and phi-2 chat used the same base model but a different prompt template. For evaluation and rating details see [model_tests.py](model_tests.py) and [scripts/03_rank_output.py](scripts/03_rank_output.py).

## License

`ragga` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
