from __future__ import annotations

from app.llm import _is_retryable
from deploy import entrypoint
from openai import RateLimitError
import httpx


def test_entrypoint_renders_local_api_key_pool(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("LLM_API_KEY", "key-one")
    monkeypatch.setenv("LLM_API_KEYS", "key-one,key-two\nkey-three")
    rendered = entrypoint.render_llm_config()
    assert 'api_keys = ["key-one", "key-two", "key-three"]' in rendered


def test_rate_limit_retry_requires_a_key_pool(monkeypatch):
    error = RateLimitError(
        "quota exceeded",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.test")),
        body={},
    )
    monkeypatch.delenv("LLM_API_KEYS", raising=False)
    assert _is_retryable(error) is False
    monkeypatch.setenv("LLM_API_KEYS", "key-one,key-two")
    assert _is_retryable(error) is True
