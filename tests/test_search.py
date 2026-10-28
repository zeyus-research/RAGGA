import pytest

from ragga.web import search as search_mod
from ragga.web.search import Search, SearchRetriever

HTML = """
<html><body>
<div class="result"><a class="result__a" href="https://a.example">Title A</a>
  <a class="result__snippet">Snippet A</a></div>
<div class="result"><span>sponsored, no link</span></div>
<div class="result"><a class="result__a" href="https://b.example">Title B</a></div>
<div class="result"><a class="result__a" href="https://c.example">Title C</a>
  <div class="result__snippet">Snippet C</div></div>
</body></html>
"""


class FakeResponse:
    text = HTML

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def calls(monkeypatch):
    calls: list[dict] = []

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(search_mod.requests, "get", fake_get)
    return calls


def test_parses_results_and_skips_malformed(calls):
    docs = Search(max_results=5).get_results("what & why?")
    assert [d.metadata["url"] for d in docs] == ["https://a.example", "https://b.example", "https://c.example"]
    assert docs[0].page_content == "Title A\nSnippet A"
    assert docs[1].page_content == "Title B\n"
    # the query is passed as a param so requests URL-encodes it
    assert calls[0]["params"] == {"q": "what & why?"}


def test_max_results(calls):
    assert len(Search(max_results=2).get_results("q")) == 2


def test_cache_hits_and_eviction(calls):
    search = Search()
    search._cache_max_size = 2
    search.get_results("one")
    search.get_results("one")
    assert len(calls) == 1
    search.get_results("two")
    search.get_results("three")
    assert list(search._result_cache) == ["two", "three"]


def test_retriever_invoke(calls):
    retriever = SearchRetriever(search=Search(max_results=1))
    docs = retriever.invoke("anything")
    assert len(docs) == 1
