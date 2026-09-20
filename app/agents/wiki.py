"""Wiki research agent: enriches Character Bible from public fandom wikis.

Only reads public pages; never trusts wiki text as instructions. Results are
stored as research notes and merged into character profiles with source links.
"""
from __future__ import annotations

import httpx
import re
from html import unescape

WIKI_SEARCH = "https://warhammer40k.fandom.com/api.php?action=opensearch&search={query}&limit=1&format=json"
WIKI_PAGE = "https://warhammer40k.fandom.com/api.php?action=query&prop=extracts&explaintext=1&titles={title}&format=json"
UA = {"User-Agent": "AudiobookStudio/1.0 (character research)"}


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


class WikiResearcher:
    """Fetches short public summaries for fictional characters."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.cache: dict[str, dict] = {}

    @staticmethod
    def extract_age(summary: str) -> str:
        """Guess an audible age band from a wiki summary."""
        low = summary.lower()
        if any(w in low for w in ("child", "boy", "girl", "подрост", "ребён", "childhood")):
            return "child"
        if any(w in low for w in ("young", "youth", "юнош", "молод", "cadet", "кадет", "neophyte")):
            return "young_adult"
        if any(w in low for w in ("ancient", "immortal", "бессмерт", "primarch", "примарх", "emperor", "император", "perpetual")):
            # Primarchs/Emperor: canonically ancient but sound like vital 30-40 y.o.
            return "middle_aged"
        if any(w in low for w in ("veteran", "old", "elder", "ancient", "стар", "ветеран")):
            return "elderly"
        return "adult"

    def search(self, name: str) -> dict:
        key = name.strip().lower()
        if key in self.cache:
            return self.cache[key]
        result = {"name": name, "title": None, "url": None, "summary": None, "source": "warhammer40k.fandom.com"}
        try:
            with httpx.Client(timeout=self.timeout, headers=UA, follow_redirects=True) as client:
                sr = client.get(WIKI_SEARCH.format(query=httpx.QueryParams({"q": name}).get("q", name)))
                sr = client.get("https://warhammer40k.fandom.com/api.php", params={
                    "action": "opensearch", "search": name, "limit": 1, "format": "json"})
                data = sr.json()
                if data and len(data) > 1 and data[1]:
                    title = data[1][0]
                    url = data[3][0] if len(data) > 3 and data[3] else None
                    pr = client.get("https://warhammer40k.fandom.com/api.php", params={
                        "action": "query", "prop": "extracts", "explaintext": 1,
                        "titles": title, "format": "json"})
                    pages = pr.json().get("query", {}).get("pages", {})
                    extract = ""
                    for page in pages.values():
                        extract = strip_tags(page.get("extract", ""))[:1200]
                        break
                    if extract:
                        result.update(title=title, url=url, summary=extract,
                                      age_band=self.extract_age(extract))
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        self.cache[key] = result
        return result
