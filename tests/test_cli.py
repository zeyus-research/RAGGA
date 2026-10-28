from pathlib import Path

import pytest
import requests

from ragga.cli import COMMANDS, build_config, handle_input, main, parse_args, user_cache_dir


def test_defaults_without_options():
    conf = build_config(parse_args([]))
    assert conf["generator"]["llm_path"] is None
    assert conf["generator"]["gpu_layers"] == -1
    assert conf["retriever"]["search"] == "hybrid"


def test_config_file_then_options_override(tmp_path):
    config = tmp_path / "my.yaml"
    config.write_text("generator:\n  max_tokens: 99\n  temperature: 0.1\ndataset:\n  path: from-file\n")
    conf = build_config(
        parse_args(["-c", str(config), "-m", "~/m.gguf", "-n", "notes", "--temperature", "0.5", "--cache-dir", "c"])
    )
    assert conf["generator"]["max_tokens"] == 99  # from the file
    assert conf["generator"]["temperature"] == 0.5  # option beats the file
    assert conf["generator"]["llm_path"] == str(Path("~/m.gguf").expanduser())
    assert conf["dataset"]["path"] == "notes"
    assert conf["encoder"]["cache_dir"] == str(Path("c") / "embeddings")
    assert conf["retriever"]["cache_dir"].startswith(str(Path("c") / "index"))
    assert conf["generator"]["context_length"] == 8192  # untouched default


def test_missing_model_and_notes_are_reported(tmp_path, capsys):
    assert main([]) == 2
    assert "No model given" in capsys.readouterr().err
    model = tmp_path / "m.gguf"
    assert main(["-m", str(model)]) == 2
    assert "Model file not found" in capsys.readouterr().err
    model.write_bytes(b"")
    assert main(["-m", str(model), "-n", str(tmp_path / "nope")]) == 2
    assert "Notes folder not found" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith("ragga ")


def test_cache_defaults_to_user_cache_dir_with_an_index_per_notes_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    a = build_config(parse_args(["-n", str(tmp_path / "a")]))
    b = build_config(parse_args(["-n", str(tmp_path / "b")]))
    root = user_cache_dir()
    assert a["encoder"]["cache_dir"] == b["encoder"]["cache_dir"] == str(root / "embeddings")
    assert a["retriever"]["cache_dir"] != b["retriever"]["cache_dir"]
    assert Path(a["retriever"]["cache_dir"]).parent == root / "index"


def test_config_file_cache_dir_beats_default_but_not_option(tmp_path):
    config = tmp_path / "my.yaml"
    config.write_text("encoder:\n  cache_dir: mine\nretriever:\n  cache_dir: mine\n")
    assert build_config(parse_args(["-c", str(config)]))["encoder"]["cache_dir"] == "mine"
    conf = build_config(parse_args(["-c", str(config), "--cache-dir", "opt"]))
    assert conf["encoder"]["cache_dir"] == str(Path("opt") / "embeddings")


def test_no_cache_keeps_embeddings_in_memory(tmp_path):
    conf = build_config(parse_args(["--no-cache"]), temp_dir=str(tmp_path))
    assert conf["encoder"]["persistent_cache"] is False
    assert conf["retriever"]["cache_dir"] == str(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["--no-cache", "--cache-dir", "x"])


def test_history_tokens_option():
    assert build_config(parse_args([]))["generator"]["history_tokens"] == 2048
    assert build_config(parse_args(["--history-tokens", "0"]))["generator"]["history_tokens"] == 0


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.last_docs = None
        self.forgotten_turns = 0
        self.history = []

    def get_answer_stream(self, question, *, web=False):
        self.calls.append(("ask", question, web))
        return iter(["answer"])

    def continue_stream(self):
        msg = "No answer to continue"
        raise ValueError(msg)

    def reset(self):
        self.calls.append(("reset",))


@pytest.mark.parametrize("line", ["exit", "QUIT", "/exit", "/quit"])
def test_exit_commands(line):
    assert handle_input(line, FakeGenerator(), web=False) is False


def test_questions_and_commands(capsys):
    generator = FakeGenerator()
    assert handle_input("what is /web?", generator, web=False)
    assert handle_input("/web  cats ", generator, web=True)
    assert handle_input("/NEW", generator, web=False)
    assert generator.calls == [("ask", "what is /web?", False), ("ask", "cats", True), ("reset",)]
    assert "[new conversation]" in capsys.readouterr().err


def test_command_errors_keep_chatting(capsys):
    generator = FakeGenerator()
    assert handle_input("/web cats", generator, web=False)
    assert "restart with --web" in capsys.readouterr().err
    assert handle_input("/web", generator, web=True)
    assert "usage: /web <query>" in capsys.readouterr().err
    assert handle_input("/continue", generator, web=False)
    assert "No answer to continue" in capsys.readouterr().err
    assert handle_input("/frobnicate", generator, web=False)
    assert "unknown command /frobnicate" in capsys.readouterr().err
    assert generator.calls == []


def test_web_search_failure_keeps_chatting(capsys):
    generator = FakeGenerator()

    def failing(question, *, web=False):
        def stream():
            msg = "offline"
            raise requests.ConnectionError(msg)
            yield

        return stream()

    generator.get_answer_stream = failing
    assert handle_input("/web cats", generator, web=True)
    assert "web search failed: offline" in capsys.readouterr().err


def test_help_lists_every_command(capsys):
    handle_input("/help", FakeGenerator(), web=False)
    err = capsys.readouterr().err
    assert all(name in err for name in COMMANDS)


def test_web_options():
    conf = build_config(parse_args([]))
    assert conf["web"]["enabled"] is False
    assert conf["web"]["engine"] == "duckduckgo"
    assert build_config(parse_args(["--web"]))["web"]["enabled"] is True
    conf = build_config(parse_args(["--searxng", "http://localhost:8888"]))
    assert conf["web"] == {**conf["web"], "enabled": True, "engine": "searxng", "searxng_url": "http://localhost:8888"}


def test_searxng_without_url_is_reported_before_loading(tmp_path, capsys):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    config = tmp_path / "my.yaml"
    config.write_text("web:\n  enabled: true\n  engine: searxng\n")
    assert main(["-m", str(model), "-n", str(tmp_path), "-c", str(config)]) == 2
    assert "searxng_url" in capsys.readouterr().err
