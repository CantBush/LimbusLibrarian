from urllib.parse import parse_qs

import httpx

from limbus_librarian.sources.mediawiki import MediaWikiSourceConnector


def test_category_discovery_and_batched_page_fetch():
    seen: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        seen.append(query)
        if query.get("list") == ["categorymembers"]:
            title = query["cmtitle"][0]
            members = (
                [
                    {"pageid": 1, "ns": 0, "title": "Yi Sang"},
                    {"pageid": 10, "ns": 14, "title": "Category:Sinners"},
                ]
                if title == "Category:Characters"
                else [{"pageid": 2, "ns": 0, "title": "Faust"}]
            )
            return httpx.Response(200, json={"query": {"categorymembers": members}})
        pages = [
            {
                "pageid": page_id,
                "ns": 0,
                "title": "Yi Sang" if page_id == 1 else "Faust",
                "fullurl": f"https://example.test/{page_id}",
                "categories": [{"title": "Category:Sinners"}],
                "revisions": [
                    {
                        "revid": 100 + page_id,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "slots": {"main": {"content": "== Story ==\nLore text."}},
                    }
                ],
            }
            for page_id in (1, 2)
        ]
        return httpx.Response(200, json={"query": {"pages": pages}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = MediaWikiSourceConnector(
        "https://example.test/api.php",
        "Test/1.0",
        min_interval_s=0,
        categories=("Characters",),
        category_depth=1,
        client=client,
    )

    listings = connector.list_pages()
    pages = connector.fetch_pages([item["page_id"] for item in listings])

    assert [item["title"] for item in listings] == ["Faust", "Yi Sang"]
    assert [page.page_id for page in pages] == [2, 1]
    assert any(query.get("pageids") == ["2|1"] for query in seen)
    assert not any(query.get("list") == ["allpages"] for query in seen)


def test_429_uses_retry_after():
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"query": {"categorymembers": []}})

    connector = MediaWikiSourceConnector(
        "https://example.test/api.php",
        "Test/1.0",
        min_interval_s=0,
        categories=("Lore",),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )

    assert connector.list_pages() == []
    assert attempts == 2
    assert sleeps == [3.0]


def test_recentchanges_is_paginated_and_marks_deletions():
    seen: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        seen.append(query)
        if "rccontinue" not in query:
            return httpx.Response(
                200,
                json={
                    "continue": {"rccontinue": "next"},
                    "query": {
                        "recentchanges": [
                            {
                                "type": "edit",
                                "pageid": 1,
                                "title": "Yi Sang",
                                "revid": 102,
                                "timestamp": "2026-02-01T00:00:00Z",
                            }
                        ]
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "query": {
                    "recentchanges": [
                        {
                            "type": "log",
                            "logtype": "delete",
                            "pageid": 0,
                            "title": "Old Page",
                            "timestamp": "2026-02-01T01:00:00Z",
                        }
                    ]
                }
            },
        )

    connector = MediaWikiSourceConnector(
        "https://example.test/api.php",
        "Test/1.0",
        min_interval_s=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    changes = connector.list_recent_changes("2026-01-01T00:00:00Z")

    assert [change["deleted"] for change in changes] == [False, True]
    assert seen[0]["rcstart"] == ["2026-01-01T00:00:00Z"]
    assert seen[1]["rccontinue"] == ["next"]
