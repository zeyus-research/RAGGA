from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragga import DATASET_FORMATS, Config, MarkdownDataset, NoteDataset, TextDataset, load_dataset


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


def test_markdown_extensions_override(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nmarkdown\n")
    (tmp_path / "b.markdown").write_text("# B\n\nalso markdown\n")
    (tmp_path / "c.MD").write_text("# C\n\nupper case\n")
    default = MarkdownDataset(dataset_config(tmp_path)).documents
    assert sorted(d.metadata["title"] for d in default) == ["a", "b", "c"]
    only_md = MarkdownDataset(dataset_config(tmp_path, extensions=["md"])).documents
    assert sorted(d.metadata["title"] for d in only_md) == ["a", "c"]


def test_text_dataset_reads_files_as_is(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "types.ml").write_text("(* Notes on GADTs *)\ntype _ t = Int : int t\n")
    (tmp_path / "sig.mli").write_text("val f : int -> int\n")
    (tmp_path / "empty.ml").write_text("  \n")
    (tmp_path / "readme.md").write_text("# not loaded\n")
    docs = TextDataset(dataset_config(tmp_path, extensions=[".ml", ".mli"])).documents
    assert {d.metadata["title"]: d.metadata["folder"] for d in docs} == {"sig": "", "types": "sub"}
    assert any(d.page_content.startswith("(* Notes on GADTs *)") for d in docs)


def test_load_dataset_by_format(tmp_path):
    assert type(load_dataset(dataset_config(tmp_path))) is MarkdownDataset
    assert type(load_dataset(dataset_config(tmp_path, format="text"))) is TextDataset
    with pytest.raises(ValueError, match=r"unknown dataset\.format"):
        load_dataset(dataset_config(tmp_path, format="org"))


def test_custom_note_dataset(tmp_path, monkeypatch):
    class UpperDataset(NoteDataset):
        extensions = (".org",)

        def _load_file(self, file, root):
            return [Document(page_content=file.read_text().upper(), metadata=self._source_metadata(file, root))]

    monkeypatch.setitem(DATASET_FORMATS, "org", UpperDataset)
    (tmp_path / "todo.org").write_text("* write notes\n")
    docs = load_dataset(dataset_config(tmp_path, format="org")).documents
    assert [d.page_content for d in docs] == ["* WRITE NOTES\n"]
    assert repr(UpperDataset(dataset_config(tmp_path))) == f"UpperDataset({tmp_path})"
