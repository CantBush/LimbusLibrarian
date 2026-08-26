from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx

from limbus_librarian.sources import RawPage


class MediaWikiSourceConnector:
    """Polite MediaWiki Action API client. Prefer dump/fixtures for tests and CI."""

    source_id = "limbuscompany_wiki"

    def __init__(
        self,
        api_url: str,
        user_agent: str,
        min_interval_s: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self._last_request = 0.0

    def _get(self, params: dict) -> dict:
        elapsed = time.monotonic() - self._last_request
        wait = self.min_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        params = {**params, "format": "json"}
        with httpx.Client(headers={"User-Agent": self.user_agent}, timeout=30) as client:
            response = client.get(self.api_url, params=params)
            response.raise_for_status()
            self._last_request = time.monotonic()
            return response.json()

    def list_pages(self) -> list[dict]:
        pages: list[dict] = []
        apcontinue: str | None = None
        while True:
            params: dict = {
                "action": "query",
                "list": "allpages",
                "apnamespace": 0,
                "aplimit": "max",
                "apfilterredir": "nonredirects",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue
            data = self._get(params)
            for item in data.get("query", {}).get("allpages", []):
                pages.append(
                    {
                        "page_id": item["pageid"],
                        "title": item["title"],
                        "namespace": item.get("ns", 0),
                    }
                )
            apcontinue = data.get("continue", {}).get("apcontinue")
            if not apcontinue:
                break
        return pages

    def fetch_page(self, page_id: int) -> RawPage:
        data = self._get(
            {
                "action": "query",
                "pageids": page_id,
                "prop": "revisions|categories|info",
                "rvslots": "main",
                "rvprop": "ids|timestamp|content",
                "inprop": "url",
                "cllimit": "max",
            }
        )
        page = next(iter(data.get("query", {}).get("pages", {}).values()))
        rev = page["revisions"][0]
        wikitext = rev["slots"]["main"].get("*") or rev["slots"]["main"].get("content", "")
        categories = [
            c.get("title", "").removeprefix("Category:")
            for c in page.get("categories", [])
        ]
        return RawPage(
            source_id=self.source_id,
            page_id=page["pageid"],
            revision_id=rev["revid"],
            title=page["title"],
            url=page.get("fullurl")
            or f"https://limbuscompany.wiki.gg/wiki/{page['title'].replace(' ', '_')}",
            namespace=page.get("ns", 0),
            wikitext=wikitext,
            categories=categories,
            last_modified=rev.get("timestamp"),
            retrieved_at=datetime.now(UTC).isoformat(),
        )
