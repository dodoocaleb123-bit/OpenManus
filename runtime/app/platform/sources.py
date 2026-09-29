from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def extract_sources(evidence: Any) -> list[dict[str, Any]]:
    """Extract bounded, deduplicated source records from nested tool evidence."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            explicit_url = value.get("url") or value.get("source_url")
            if isinstance(explicit_url, str):
                add(explicit_url, value)
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            for url in _URL_RE.findall(value):
                add(url, {})

    def add(raw_url: str, metadata: dict[str, Any]) -> None:
        url = raw_url.rstrip(".,);]}")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen:
            return
        seen.add(url)
        record: dict[str, Any] = {
            "url": url,
            "host": parsed.netloc,
            "path": parsed.path or "/",
            "title": metadata.get("title") or metadata.get("page_title"),
            "status": metadata.get("status") or metadata.get("http_status"),
            "accessed": metadata.get("accessed") or metadata.get("retrieved_at"),
            "excerpt": str(metadata.get("excerpt") or metadata.get("text") or "")[:500],
        }
        found.append({key: value for key, value in record.items() if value not in (None, "")})

    walk(evidence)
    return found[:100]


def citation_records(evidence: Any) -> list[dict[str, Any]]:
    """Return citation-ready source records with retrieval metadata."""
    accessed = datetime.now(timezone.utc).isoformat()
    return [
        {
            **source,
            "retrieved_at": source.get("accessed") or accessed,
            "confidence": "source_recorded",
        }
        for source in extract_sources(evidence)
    ]
