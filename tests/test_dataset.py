import pytest

from ragga import Config, MarkdownDataset


def dataset_config(path) -> Config:
    return Config(config={"dataset": {"path": str(path), "show_progress": False, "use_multithreading": False}})


def test_loads_markdown(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "one.md").write_text("# Title\n\nSome notes about sonification.\n")
    (tmp_path / "sub" / "two.md").write_text("# Other\n\nMore notes.\n")
    (tmp_path / "ignored.txt").write_text("not markdown")
    docs = MarkdownDataset(dataset_config(tmp_path)).documents
    assert len(docs) == 2
    assert any("sonification" in d.page_content for d in docs)
    assert all(d.metadata["source"].endswith(".md") for d in docs)


def test_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _ = MarkdownDataset(dataset_config(tmp_path / "nope")).documents


def test_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No documents"):
        _ = MarkdownDataset(dataset_config(tmp_path)).documents
