from pathlib import Path

import pytest

from ragga import Config, MarkdownDataset


def dataset_config(path, **extra) -> Config:
    return Config(config={"dataset": {"path": str(path), "show_progress": False, "use_multithreading": False, **extra}})


def test_loads_markdown(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "one.md").write_text("# Title\n\nSome notes about sonification.\n")
    (tmp_path / "sub" / "two.md").write_text("# Other\n\nMore notes.\n")
    (tmp_path / "ignored.txt").write_text("not markdown")
    docs = MarkdownDataset(dataset_config(tmp_path)).documents
    assert len(docs) == 2
    assert any("sonification" in d.page_content for d in docs)
    assert all(d.metadata["source"].endswith(".md") for d in docs)
    meta = {d.metadata["title"]: d.metadata["folder"] for d in docs}
    assert meta == {"one": "", "two": "sub"}


def test_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _ = MarkdownDataset(dataset_config(tmp_path / "nope")).documents


def test_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No documents"):
        _ = MarkdownDataset(dataset_config(tmp_path)).documents


def test_skips_hidden_and_respects_recursive(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.md").write_text("# hidden\n")
    (tmp_path / ".draft.md").write_text("# hidden\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.md").write_text("# Deep\n\nnested\n")
    (tmp_path / "top.md").write_text("# Top\n\nvisible\n")
    recursive = MarkdownDataset(dataset_config(tmp_path, use_multithreading=True)).documents
    assert sorted(Path(d.metadata["source"]).name for d in recursive) == ["deep.md", "top.md"]
    flat = MarkdownDataset(dataset_config(tmp_path, recursive=False)).documents
    assert [Path(d.metadata["source"]).name for d in flat] == ["top.md"]


def test_elements_mode(tmp_path):
    (tmp_path / "a.md").write_text("# Heading\n\nFirst paragraph.\n\nSecond paragraph.\n")
    conf = dataset_config(tmp_path, loader_kwargs={"mode": "elements", "unstructured_kwargs": {}})
    docs = MarkdownDataset(conf).documents
    assert len(docs) == 3
    assert docs[0].page_content == "Heading"
    assert docs[0].metadata["category"] == "Title"
    assert docs[0].metadata["source"].endswith("a.md")
    assert "element_id" in docs[0].metadata
