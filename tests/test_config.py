from types import MappingProxyType

import pytest

from ragga import Config
from ragga.core.config import Configurable, deep_update


class Thing(Configurable):
    _config_key = "thing"
    _default_config = MappingProxyType({"a": 1, "b": {"c": 2}})


def test_load_from_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("encoder:\n  model_path: foo\n  dims: !!python/tuple [1, 2]\n")
    conf = Config(config_path=path)
    assert conf["encoder"]["model_path"] == "foo"
    assert conf["encoder"]["dims"] == (1, 2)
    assert "encoder" in conf


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Config(config_path=tmp_path / "nope.yaml")


def test_repo_config_loads():
    conf = Config(config_path="config.yaml")
    assert {"encoder", "retriever", "generator", "dataset", "prompt"} <= set(conf.config)


def test_deep_update_merges_nested_without_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_update(base, {"a": {"b": 10}}, {"e": 5})
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_merge():
    conf = Config(config={"x": {"y": 1, "z": 2}})
    conf.merge({"x": {"y": 3}})
    assert conf["x"] == {"y": 3, "z": 2}


def test_configurable_fills_defaults():
    conf = Config(config={"thing": {"a": 5}})
    thing = Thing(conf)
    assert conf["thing"] == {"a": 5, "b": {"c": 2}}
    assert thing.key == "thing"


def test_configurable_creates_section():
    conf = Config(config={})
    Thing(conf)
    assert conf["thing"] == {"a": 1, "b": {"c": 2}}


def test_add_config():
    thing = Thing(Config(config={}))
    thing.add_config("a", 9)
    thing.add_config({"z": True})
    assert thing.config["thing"]["a"] == 9
    assert thing.config["thing"]["z"] is True
    with pytest.raises(TypeError):
        thing.add_config("a", None)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        thing.add_config(3)  # type: ignore[call-overload]
