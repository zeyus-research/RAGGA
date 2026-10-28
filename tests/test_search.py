from typing import ClassVar

import pytest

from ragga import Config
from ragga.web import search as search_mod
from ragga.web.search import Search, SearchRetriever, SearXNGSearch, web_retriever

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

REDIRECTS = """
<div class="result result--ad"><a class="result__a" href="//duckduckgo.com/y.js?ad_domain=x">Marked ad</a></div>
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad">
  Unmarked ad</a></div>
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&rut=abc">
  Python</a></div>
"""


class FakeResponse:
    text = HTML
    status_code = 200
    payload: ClassVar[dict] = {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


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


def test_unwraps_redirects_and_skips_ads(calls, monkeypatch):
    monkeypatch.setattr(FakeResponse, "text", REDIRECTS)
    docs = Search().get_results("python")
    assert [d.metadata["url"] for d in docs] == ["https://www.python.org/"]


def test_searxng(calls, monkeypatch):
    results = [
        {"title": "A", "url": "https://a.example", "content": "Snippet A"},
        {"title": "No url"},
        {"title": "B", "url": "https://b.example", "content": None},
        {"title": "C", "url": "https://c.example"},
    ]
    monkeypatch.setattr(FakeResponse, "payload", {"results": results})
    docs = SearXNGSearch("http://localhost:8888/", max_results=2).get_results("what & why?")
    assert [d.page_content for d in docs] == ["A\nSnippet A", "B\n"]
    assert calls[0]["url"] == "http://localhost:8888/search"
    assert calls[0]["params"] == {"q": "what & why?", "format": "json"}


def test_searxng_without_json_format(calls, monkeypatch):
    monkeypatch.setattr(FakeResponse, "status_code", 403)
    with pytest.raises(search_mod.requests.HTTPError, match=r"search\.formats"):
        SearXNGSearch("http://localhost:8888").get_results("q")


def test_web_retriever_from_config():
    assert isinstance(web_retriever(Config(config={})).search, Search)
    searx = web_retriever(Config(config={"web": {"engine": "searxng", "searxng_url": "http://s", "max_results": 3}}))
    assert isinstance(searx.search, SearXNGSearch)
    assert searx.search.max_results == 3
    with pytest.raises(ValueError, match="searxng_url"):
        web_retriever(Config(config={"web": {"engine": "searxng"}}))
    with pytest.raises(ValueError, match=r"unknown web\.engine"):
        web_retriever(Config(config={"web": {"engine": "bing"}}))
