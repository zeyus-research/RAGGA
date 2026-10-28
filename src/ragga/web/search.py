import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import bs4
import requests
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

    from ragga.core.config import Config

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0"


class WebSearch(ABC):
    """A web search engine, results are cached per query."""

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._user_agent = USER_AGENT
        self._cache_max_size = 100
        self._result_cache: OrderedDict[str, list[Document]] = OrderedDict()

    @abstractmethod
    def _results(self, question: str) -> list[Document]:
        """Search, returning up to max_results documents ("title\\nsnippet", with the page's url in the metadata)."""

    def get_results(self, question: str) -> list[Document]:
        if question in self._result_cache:
            self._result_cache.move_to_end(question)
            return self._result_cache[question]
        results = self._results(question)
        while len(self._result_cache) >= self._cache_max_size:
            self._result_cache.popitem(last=False)
        self._result_cache[question] = results
        return results


class Search(WebSearch):
    """DuckDuckGo, via its HTML (no JavaScript) page."""

    def _search(self, question: str) -> list[bs4.Tag]:
        logging.info(f"Searching DuckDuckGo for: {question}")
        response = requests.get(SEARCH_URL, params={"q": question}, timeout=5, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.select("div.result:not(.result--ad)")

    @staticmethod
    def _unwrap(url: str) -> str:
        """Result links go through a DuckDuckGo redirect (//duckduckgo.com/l/?uddg=<url>), return the real url."""
        if urlparse(url).path == "/l/" and (target := parse_qs(urlparse(url).query).get("uddg")):
            return target[0]
        return url

    def _results(self, question: str) -> list[Document]:
        results: list[Document] = []
        for result in self._search(question):
            if len(results) >= self.max_results:
                break
            link = result.select_one("a.result__a")
            snippet_el = result.select_one(".result__snippet")
            if link is None:  # malformed entries
                continue
            url = self._unwrap(str(link.get("href", "")))
            if "duckduckgo.com/y.js" in url:  # ads that aren't marked as such
                continue
            title = link.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el is not None else ""
            results.append(Document(page_content=title + "\n" + snippet, metadata={"url": url}))
        return results


DuckDuckGoSearch = Search


class SearXNGSearch(WebSearch):
    """A SearXNG instance (https://docs.searxng.org), e.g. one you host yourself, via its JSON API.

    The instance has to allow JSON results: add `json` to `search.formats` in its settings.yml.
    """

    def __init__(self, url: str, max_results: int = 5) -> None:
        super().__init__(max_results)
        self.url = url.rstrip("/")

    def _results(self, question: str) -> list[Document]:
        logging.info(f"Searching {self.url} for: {question}")
        response = requests.get(
            f"{self.url}/search",
            params={"q": question, "format": "json"},
            timeout=10,
            headers={"User-Agent": self._user_agent},
        )
        if response.status_code == 403:  # noqa: PLR2004
            msg = f"{self.url} doesn't allow JSON results, add json to search.formats in its settings.yml"
            raise requests.HTTPError(msg, response=response)
        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results", [])
        return [
            Document(page_content=f"{r.get('title', '')}\n{r.get('content') or ''}", metadata={"url": r["url"]})
            for r in results
            if r.get("url")
        ][: self.max_results]


class SearchRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    search: WebSearch

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",  # noqa: ARG002
    ) -> list[Document]:
        return self.search.get_results(query)


def web_retriever(conf: "Config") -> SearchRetriever:
    """The search engine configured in the `web` config section."""
    web = conf["web"] if "web" in conf else {}
    engine = web.get("engine", "duckduckgo")
    max_results = web.get("max_results", 5)
    if engine == "duckduckgo":
        return SearchRetriever(search=Search(max_results=max_results))
    if engine == "searxng":
        if not web.get("searxng_url"):
            msg = "web.engine is searxng, but web.searxng_url isn't set (or use --searxng URL)"
            raise ValueError(msg)
        return SearchRetriever(search=SearXNGSearch(web["searxng_url"], max_results=max_results))
    msg = f'unknown web.engine "{engine}", use "duckduckgo" or "searxng"'
    raise ValueError(msg)


WebSearchRetriever = SearchRetriever(search=Search())
