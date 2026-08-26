from __future__ import annotations

import re

import mwparserfromhell

from limbus_librarian.ingest.classify import classify_document, detect_cantos
from limbus_librarian.models import SourceDocument
from limbus_librarian.sources import RawPage

_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.M)


def parse_wikitext(wikitext: str) -> tuple[str, dict[str, str], list[str], list[str]]:
    code = mwparserfromhell.parse(wikitext)
    infobox: dict[str, str] = {}
    entities: list[str] = []
    for template in code.filter_templates():
        name = template.name.strip().lower()
        if "infobox" in name:
            for param in template.params:
                key = str(param.name).strip()
                val = param.value.strip_code().strip()
                if key and val:
                    infobox[key] = val
    for link in code.filter_wikilinks():
        target = str(link.title).split("|", 1)[0].strip()
        if target and not target.lower().startswith(("file:", "category:", "http")):
            entities.append(target)
    unique_entities = list(dict.fromkeys(entities))
    headings = [m.group(2).strip() for m in _HEADING.finditer(wikitext)]
    plain = code.strip_code()
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
    return plain, infobox, headings, unique_entities


def raw_to_document(page: RawPage, corpus_version: str) -> SourceDocument:
    plain, infobox, headings, entities = parse_wikitext(page.wikitext)
    document_type = classify_document(page.title, page.categories)
    return SourceDocument(
        doc_id=f"wiki.gg:limbuscompany_en:{page.page_id}",
        source_id=page.source_id,
        url=page.url,
        title=page.title,
        namespace=page.namespace,
        page_id=page.page_id,
        revision_id=page.revision_id,
        document_type=document_type,
        categories=page.categories,
        section_outline=headings,
        entities=entities,
        cantos=detect_cantos(page.title, page.categories, plain),
        retrieved_at=page.retrieved_at,
        last_modified=page.last_modified,
        raw_wikitext=page.wikitext,
        plain_text=plain,
        infobox=infobox,
        corpus_version=corpus_version,
        attribution_text=(
            "Sample fixture for Limbus Librarian tests. Not a wiki copy. "
            "Live wiki text is CC BY-SA 4.0 (Limbus Company Wiki / wiki.gg)."
            if page.url.startswith("https://limbuscompany.wiki.gg/") is False
            or "fixture" in page.url
            else "Text from the Limbus Company Wiki (wiki.gg), CC BY-SA 4.0."
        ),
    )
