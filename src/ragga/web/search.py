import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import bs4
import requests
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

SEARCH_URL = "https://html.duckduckgo.com/html/"


class Search:
    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0"
        self._cache_max_size = 100
        self._result_cache: OrderedDict[str, list[Document]] = OrderedDict()

    def _search(self, question: str) -> list[bs4.Tag]:
        logging.info(f"Searching for: {question}")
        response = requests.get(SEARCH_URL, params={"q": question}, timeout=5, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return soup.select("div.result")

    def get_results(self, question: str) -> list[Document]:
        if question in self._result_cache:
            self._result_cache.move_to_end(question)
            return self._result_cache[question]
        results: list[Document] = []
        for result in self._search(question):
            if len(results) >= self.max_results:
                break
            link = result.select_one("a.result__a")
            snippet_el = result.select_one(".result__snippet")
            if link is None:  # ads / malformed entries
                continue
            title = link.get_text(strip=True)
            url = str(link.get("href", ""))
            snippet = snippet_el.get_text(strip=True) if snippet_el is not None else ""
            results.append(Document(page_content=title + "\n" + snippet, metadata={"url": url}))
        while len(self._result_cache) >= self._cache_max_size:
            self._result_cache.popitem(last=False)
        self._result_cache[question] = results

        return results


class SearchRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    search: Search

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",  # noqa: ARG002
    ) -> list[Document]:
        return self.search.get_results(query)


WebSearchRetriever = SearchRetriever(search=Search())
