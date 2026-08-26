from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from limbus_librarian.sources import RawPage

DEFAULT_LORE_CATEGORIES = (
    "Characters",
    "Sinners",
    "Story",
    "Cantos",
    "Factions",
    "Abnormalities",
    "Locations",
    "Lore",
)


class MediaWikiSourceConnector:
    """Polite MediaWiki Action API client. Prefer dump/fixtures for tests and CI."""

    source_id = "limbuscompany_wiki"

    def __init__(
        self,
        api_url: str,
        user_agent: str,
        min_interval_s: float = 1.0,
        categories: tuple[str, ...] = DEFAULT_LORE_CATEGORIES,
        category_depth: int = 2,
        batch_size: int = 50,
        max_retries: int = 5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_url = api_url
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.categories = categories
        self.category_depth = category_depth
        self.batch_size = min(max(batch_size, 1), 50)
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep
        self._last_request = 0.0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": 2}
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_request
            wait = self.min_interval_s - elapsed
            if wait > 0:
                self._sleep(wait)
            try:
                if self._client is None:
                    with httpx.Client(
                        headers={"User-Agent": self.user_agent}, timeout=30
                    ) as client:
                        response = client.get(self.api_url, params=params)
                else:
                    response = self._client.get(
                        self.api_url,
                        params=params,
                        headers={"User-Agent": self.user_agent},
                        timeout=30,
                    )
                self._last_request = time.monotonic()
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    self._sleep(self._retry_delay(response, attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.RequestError:
                self._last_request = time.monotonic()
                if attempt >= self.max_retries:
                    raise
                self._sleep(min(2**attempt, 60))
        raise RuntimeError("MediaWiki request retry loop ended unexpectedly")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(retry_after)
                    return max((parsed - datetime.now(UTC)).total_seconds(), 0.0)
                except (TypeError, ValueError):
                    pass
        return min(2**attempt, 60)

    def list_pages(self) -> list[dict]:
        """Discover namespace-zero lore pages from configured category trees."""
        pages: dict[int, dict] = {}
        visited_categories: set[str] = set()
        queue = [(self._category_title(category), 0) for category in self.categories]
        while queue:
            category, depth = queue.pop(0)
            if category in visited_categories:
                continue
            visited_categories.add(category)
            cmcontinue: str | None = None
            while True:
                params: dict = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmnamespace": "0|14",
                    "cmtype": "page|subcat",
                    "cmlimit": "max",
                }
                if cmcontinue:
                    params["cmcontinue"] = cmcontinue
                data = self._get(params)
                for item in data.get("query", {}).get("categorymembers", []):
                    namespace = item.get("ns", 0)
                    if namespace == 0:
                        pages[item["pageid"]] = {
                            "page_id": item["pageid"],
                            "title": item["title"],
                            "namespace": namespace,
                        }
                    elif namespace == 14 and depth < self.category_depth:
                        queue.append((item["title"], depth + 1))
                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue:
                    break
        return sorted(pages.values(), key=lambda item: (item["title"], item["page_id"]))

    def list_recent_changes(self, since: str) -> list[dict]:
        """Return namespace-zero edits/new pages and deletion logs after ``since``."""
        changes: list[dict] = []
        continuation: dict = {}
        while True:
            data = self._get(
                {
                    "action": "query",
                    "list": "recentchanges",
                    "rcnamespace": "0",
                    "rctype": "edit|new|log",
                    "rcprop": "title|ids|timestamp|loginfo",
                    "rcstart": since,
                    "rcdir": "newer",
                    "rclimit": "max",
                    **continuation,
                }
            )
            for item in data.get("query", {}).get("recentchanges", []):
                is_delete = item.get("type") == "log" and item.get("logtype") == "delete"
                page_id = int(item.get("pageid") or 0)
                changes.append(
                    {
                        "page_id": page_id,
                        "title": str(item.get("title", "")),
                        "revision_id": int(item.get("revid") or 0),
                        "timestamp": str(item.get("timestamp", "")),
                        "deleted": is_delete,
                    }
                )
            continuation = data.get("continue", {})
            if not continuation:
                break
        return sorted(
            changes,
            key=lambda item: (
                item["timestamp"],
                item["page_id"],
                item["title"].casefold(),
                item["revision_id"],
            ),
        )

    @staticmethod
    def _category_title(category: str) -> str:
        return category if category.startswith("Category:") else f"Category:{category}"

    def fetch_page(self, page_id: int) -> RawPage:
        pages = self.fetch_pages([page_id])
        if not pages:
            raise KeyError(f"MediaWiki page {page_id} was missing")
        return pages[0]

    def fetch_pages(self, page_ids: list[int]) -> list[RawPage]:
        results: list[RawPage] = []
        for start in range(0, len(page_ids), self.batch_size):
            results.extend(self._fetch_batch(page_ids[start : start + self.batch_size]))
        return results

    def _fetch_batch(self, page_ids: list[int]) -> list[RawPage]:
        merged: dict[int, dict] = {}
        continuation: dict = {}
        while True:
            data = self._get(
                {
                    "action": "query",
                    "pageids": "|".join(str(page_id) for page_id in page_ids),
                    "prop": "revisions|categories|info",
                    "rvslots": "main",
                    "rvprop": "ids|timestamp|content",
                    "inprop": "url",
                    "cllimit": "max",
                    **continuation,
                }
            )
            raw_pages = data.get("query", {}).get("pages", [])
            if isinstance(raw_pages, dict):
                raw_pages = list(raw_pages.values())
            for page in raw_pages:
                if page.get("missing"):
                    continue
                page_id = page["pageid"]
                if page_id not in merged:
                    merged[page_id] = page
                    merged[page_id]["categories"] = list(page.get("categories", []))
                else:
                    merged[page_id]["categories"].extend(page.get("categories", []))
            continuation = data.get("continue", {})
            if not continuation:
                break
        return [self._raw_page(merged[page_id]) for page_id in page_ids if page_id in merged]

    def _raw_page(self, page: dict) -> RawPage:
        rev = page["revisions"][0]
        main = rev.get("slots", {}).get("main", rev)
        wikitext = main.get("*") or main.get("content", "")
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
