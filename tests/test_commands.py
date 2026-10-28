import pytest

from ragga.crafting.commands import extract_keywords

KNOWN = {"notes", "note", "writing", "writings"}


def test_search_command():
    command, _, what, _ = extract_keywords("search for sonification", KNOWN)
    assert command == "search"
    assert what is not None
    assert "sonification" in what


@pytest.mark.parametrize("text", ["continue", "tell me about sonification", "what is sonification?"])
def test_returns_lowercase_tuple(text):
    command, where, what, known = extract_keywords(text, KNOWN)
    assert command is None or command == command.lower()
    for part in (where, what):
        assert part is None or part == part.lower()
    assert isinstance(known, bool)


def test_known_location_detected():
    *_, known = extract_keywords("summarise my notes on philosophy of cognitive science", KNOWN)
    assert known
