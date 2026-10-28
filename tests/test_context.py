from langchain_core.documents import Document

from ragga.pipeline.generation import Generator

HEADER = "Note: Paradigm (PhD)\n\n"


def chunk(text, start, source="p.md", header=HEADER):
    return Document(page_content=header + text, metadata={"source": source, "header": header, "start_index": start})


def test_group_by_note_orders_notes_by_rank_and_chunks_by_position():
    docs = [
        chunk("second part", 100),
        Document(page_content="loose web result", metadata={}),
        chunk("first part", 0),
        chunk("other note", 0, source="o.md", header="Note: Other\n\n"),
    ]
    assert Generator._group_by_note(docs) == [
        "## Paradigm (PhD)\nfirst part\n[...]\nsecond part",
        "loose web result",
        "## Other\nother note",
    ]


def test_group_by_note_merges_overlapping_chunks():
    text = "0123456789abcdefghij"
    docs = [chunk(text[8:20], 8), chunk(text[0:12], 0)]
    assert Generator._group_by_note(docs) == [f"## Paradigm (PhD)\n{text}"]
