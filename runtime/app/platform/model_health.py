"""Safe, non-generative health checks for configured OpenAI-compatible/Ollama roles."""
from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import config
from app.platform.model_profiles import profiles


def _models_url(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path + "/api/tags", "", ""))


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


async def check_model_role(role: str, *, timeout: float = 8.0) -> dict[str, Any]:
    settings = config.llm.get(role)
    if settings is None and role == "heavy_coding":
        settings = config.llm.get("default")
    if settings is None:
        return {"role": role, "configured": False, "reachable": False, "error": "role_not_configured"}
    url = _models_url(settings.base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
        names = {str(item.get("name", "")) for item in payload.get("models", []) if isinstance(item, dict)}
        model = settings.model
        # A loaded 1.5B model does not satisfy a configured 7B role merely
        # because both belong to the same Ollama model family.
        present = model in names or (":" not in model and f"{model}:latest" in names)
        return {"role": role, "model": model, "endpoint": _safe_url(url), "configured": True, "reachable": True, "model_present": present, "error": None if present else "model_tag_not_listed"}
    except Exception as exc:
        return {"role": role, "model": settings.model, "endpoint": _safe_url(url), "configured": True, "reachable": False, "model_present": False, "error": str(exc)[:300]}


async def check_all_model_roles() -> list[dict[str, Any]]:
    roles = profiles()
    # Health checks are lightweight and bounded; model inference remains sequential.
    results = []
    for profile in roles:
        result = await check_model_role(profile.config_name)
        result["role"] = profile.role
        result["config_name"] = profile.config_name
        result["capability"] = profile.capability
        results.append(result)
    return results
