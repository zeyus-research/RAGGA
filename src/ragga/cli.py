"""Command line chat with your notes: `ragga -m model.gguf -n ~/notes`"""

import argparse
import hashlib
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from huggingface_hub.utils import logging as hf_logging
from transformers.utils import logging as transformers_logging

from ragga import Encoder, Generator, MarkdownDataset, VectorDatabase, WebSearchRetriever
from ragga.__about__ import __version__
from ragga.core.config import DEFAULT_CONFIG_PATH, Config
from ragga.crafting.prompt import GGUFChatPrompt, Llama2ChatPrompt, Phi2ChatPrompt, Prompt, TinyLlamaChatPrompt

PROMPTS: dict[str, type[Prompt]] = {
    "auto": GGUFChatPrompt,
    "tinyllama": TinyLlamaChatPrompt,
    "llama2": Llama2ChatPrompt,
    "phi2": Phi2ChatPrompt,
}

HELP = """Ask anything about your notes. Commands:
  search <query>   answer from a web search instead of your notes (needs --web)
  continue         continue the last answer
  redo             answer the last question again
  exit             quit (or Ctrl-D)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ragga",
        description="Chat with your markdown notes using a local LLM (llama.cpp).",
        epilog=f"Defaults come from {DEFAULT_CONFIG_PATH}. Settings in --config override them, options override both.",
    )
    parser.add_argument("-m", "--model", type=Path, help="GGUF chat model (generator.llm_path)")
    parser.add_argument("-n", "--notes", type=Path, help="folder with markdown notes (dataset.path)")
    parser.add_argument("-c", "--config", type=Path, help="YAML config file, merged over the defaults")
    parser.add_argument("-q", "--question", help="answer one question and exit, instead of chatting")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--cache-dir", type=Path, help=f"where embeddings and search indexes are stored (default: {user_cache_dir()})"
    )
    cache.add_argument(
        "--no-cache", action="store_true", help="keep nothing on disk (every run embeds all notes again: slow)"
    )
    parser.add_argument("--gpu-layers", type=int, help="LLM layers to offload to the GPU, -1 for all")
    parser.add_argument("--context-length", type=int, help="LLM context size in tokens")
    parser.add_argument("--max-tokens", type=int, help="maximum answer length in tokens")
    parser.add_argument("--temperature", type=float, help="sampling temperature")
    parser.add_argument("--num-docs", type=int, help="note chunks to give the LLM")
    parser.add_argument("--embedding-model", help="sentence-transformers model for searching notes")
    parser.add_argument("--index-device", help='device for embedding notes: "auto", "cpu", "cuda", "mps"')
    parser.add_argument(
        "--prompt",
        choices=PROMPTS,
        default="auto",
        help="prompt format; auto uses the chat template embedded in the model (default: %(default)s)",
    )
    parser.add_argument("--web", action="store_true", help='allow "search <query>" to search the web (DuckDuckGo)')
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def user_cache_dir() -> Path:
    """The platform's per-user cache folder for RAGGA."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "ragga" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ragga"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ragga"


def cache_layout(root: Path, notes: str | Path) -> dict[str, dict[str, Any]]:
    """Embeddings are shared (they're keyed by content), each notes folder gets its own search index."""
    notes_id = hashlib.sha256(str(Path(notes).expanduser().resolve()).encode()).hexdigest()[:12]
    return {
        "encoder": {"cache_dir": str(root / "embeddings")},
        "retriever": {"cache_dir": str(root / "index" / notes_id)},
    }


def build_config(args: argparse.Namespace, temp_dir: str | None = None) -> Config:
    """Defaults, then the --config file, then command line options.

    Unless a config file or option sets them, caches go to the user cache folder. With --no-cache, embeddings stay in
    memory and the search index goes to `temp_dir`.
    """
    conf = Config(config_path=DEFAULT_CONFIG_PATH)
    user = Config(config_path=args.config).config if args.config is not None else {}
    notes = args.notes or user.get("dataset", {}).get("path") or conf["dataset"]["path"]
    conf.merge(cache_layout(args.cache_dir.expanduser() if args.cache_dir else user_cache_dir(), notes))
    conf.merge(user)
    if args.cache_dir is not None:  # the option beats the config file
        conf.merge(cache_layout(args.cache_dir.expanduser(), notes))
    if args.no_cache:
        conf.merge({"encoder": {"persistent_cache": False}, "retriever": {"cache_dir": temp_dir}})
    overrides: dict[str, dict[str, Any]] = {"generator": {}, "dataset": {}, "encoder": {}, "retriever": {}}
    options = {
        ("generator", "llm_path"): args.model,
        ("generator", "gpu_layers"): args.gpu_layers,
        ("generator", "context_length"): args.context_length,
        ("generator", "max_tokens"): args.max_tokens,
        ("generator", "temperature"): args.temperature,
        ("dataset", "path"): args.notes,
        ("encoder", "model_path"): args.embedding_model,
        ("encoder", "index_device"): args.index_device,
        ("retriever", "num_docs"): args.num_docs,
    }
    for (section, key), value in options.items():
        if value is not None:
            overrides[section][key] = str(value.expanduser()) if isinstance(value, Path) else value
    conf.merge(overrides)
    return conf


def _check_paths(conf: Config) -> str | None:
    model = conf["generator"].get("llm_path")
    if not model:
        return "No model given: use -m/--model path/to/model.gguf (or set generator.llm_path in a config file)."
    if not Path(model).expanduser().is_file():
        return f"Model file not found: {model}"
    if not Path(conf["dataset"]["path"]).expanduser().is_dir():
        return f"Notes folder not found: {conf['dataset']['path']} (use -n/--notes)"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)])
    if not args.verbose:
        # "unauthenticated requests to the HF Hub" on every start, weight loading progress bars,
        # and torch's "attention on AMD GPU is still experimental"
        hf_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        warnings.filterwarnings("ignore", message=r".*is still experimental", category=UserWarning)
    with tempfile.TemporaryDirectory(prefix="ragga-") as temp_dir:
        return _run(args, temp_dir)


def _run(args: argparse.Namespace, temp_dir: str) -> int:
    conf = build_config(args, temp_dir)
    if error := _check_paths(conf):
        print(f"ragga: {error}", file=sys.stderr)  # noqa: T201
        return 2

    print("Loading notes...", file=sys.stderr)  # noqa: T201
    documents = MarkdownDataset(conf).documents
    with VectorDatabase(conf, Encoder(conf)) as db:
        db.documents = documents  # only new or changed notes are embedded
        print("Loading model...", file=sys.stderr)  # noqa: T201
        generator = Generator(conf, PROMPTS[args.prompt](conf), db, websearch=WebSearchRetriever if args.web else None)
        generator.subscribe(lambda token: print(token, end="", flush=True))  # noqa: T201

        def answer(question: str) -> None:
            for _ in generator.get_answer_stream(question):
                pass
            sources = dict.fromkeys(d.metadata["title"] for d in generator.last_docs or [] if "title" in d.metadata)
            print(f"\n\nSources: {', '.join(sources)}" if sources else "")  # noqa: T201

        if args.question:
            answer(args.question)
            return 0

        print(HELP, file=sys.stderr)  # noqa: T201
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)  # noqa: T201
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                continue
            try:
                answer(question)
            except KeyboardInterrupt:  # stop this answer, keep chatting
                print("\n[stopped]", file=sys.stderr)  # noqa: T201
            except ValueError as e:  # e.g. "continue" before any answer
                print(f"ragga: {e}", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    sys.exit(main())
