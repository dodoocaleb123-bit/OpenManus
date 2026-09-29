from __future__ import annotations

import asyncio
import re
import urllib.request
from typing import Any

from app.platform.specialists import bounded_gather

_MAX_BODY = 300_000

async def _fetch(url: str) -> dict[str, Any]:
    def read() -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "OpenManus-Research/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(_MAX_BODY)
            text = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", text).strip()
            return {"url": url, "status": response.status, "title": text[:160], "excerpt": text[:2000]}
    try:
        return await asyncio.to_thread(read)
    except Exception as exc:
        return {"url": url, "error": str(exc)[:300]}

async def parallel_research(urls: list[str], *, limit: int = 3) -> list[dict[str, Any]]:
    clean = list(dict.fromkeys(url for url in urls if isinstance(url, str) and url.startswith(("http://", "https://"))))[:20]
    return await bounded_gather(clean, _fetch, limit=limit)
