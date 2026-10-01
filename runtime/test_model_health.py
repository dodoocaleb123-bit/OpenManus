from __future__ import annotations

import asyncio

import httpx

from app.config import config
from app.platform import model_health
from deploy.check_multimodel import _local_host, _ollama_generate_url


def test_missing_specialist_is_not_silently_replaced_by_default(monkeypatch):
    monkeypatch.delitem(config.llm, "research", raising=False)
    result = asyncio.run(model_health.check_model_role("research"))
    assert result == {
        "role": "research", "configured": False, "reachable": False,
        "error": "role_not_configured",
    }


def test_model_health_requires_exact_local_ollama_tag(monkeypatch):
    settings = (config.llm.get("reasoning") or config.llm["default"]).model_copy(
        update={"model": "deepseek-r1:7b", "base_url": "http://model.test:11434/v1"}
    )
    monkeypatch.setitem(config.llm, "reasoning", settings)
    tag = {"name": "deepseek-r1:1.5b"}
    async_client = httpx.AsyncClient

    def client(*args, **kwargs):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [tag.copy()]}))
        return async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(model_health.httpx, "AsyncClient", client)
    wrong_model = asyncio.run(model_health.check_model_role("reasoning"))
    assert wrong_model["reachable"] is True
    assert wrong_model["model_present"] is False
    assert wrong_model["error"] == "model_tag_not_listed"

    tag["name"] = "deepseek-r1:7b"
    correct_model = asyncio.run(model_health.check_model_role("reasoning"))
    assert correct_model["model_present"] is True
    assert correct_model["error"] is None


def test_five_model_probe_only_calls_local_ollama_and_has_correct_unload_url():
    assert _local_host("http://host.docker.internal:11434/v1")
    assert _local_host("http://127.0.0.1:11434/v1")
    assert _local_host("http://192.168.1.12:11434/v1")
    assert not _local_host("https://api.example.com/v1")
    assert _ollama_generate_url("http://host.docker.internal:11434/v1") == (
        "http://host.docker.internal:11434/api/generate"
    )
