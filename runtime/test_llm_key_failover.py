from __future__ import annotations

from app.llm import _is_retryable
from app.llm import LLM
from deploy import entrypoint
from openai import RateLimitError
import httpx
from types import SimpleNamespace


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
    # Retry is bounded even with one key so transient provider throttling can recover.
    assert _is_retryable(error) is True
    monkeypatch.setenv("LLM_API_KEYS", "key-one,key-two")
    assert _is_retryable(error) is True


def test_provider_key_pools_are_exhausted_before_fallback(monkeypatch):
    llm = object.__new__(LLM)
    llm.api_keys = ["vision-one", "vision-two", "vision-three"]
    llm._api_key_index = 0
    llm.api_key = llm.api_keys[0]
    llm._using_fallback = False
    llm._fallback_config = SimpleNamespace(
        model="ollama-model",
        max_tokens=4096,
        temperature=0.0,
        api_type="openai",
        api_version="",
        base_url="http://ollama:11434/v1",
        api_key="ollama",
        api_keys=["ollama", "ollama-two"],
    )
    monkeypatch.setattr(LLM, "_make_client", lambda self, key: key)

    llm._rotate_api_key()
    assert (llm.api_key, llm._using_fallback) == ("vision-two", False)
    llm._rotate_api_key()
    assert (llm.api_key, llm._using_fallback) == ("vision-three", False)
    llm._rotate_api_key()
    assert (llm.api_key, llm._using_fallback) == ("ollama", True)
    llm._rotate_api_key()
    assert llm.api_key == "ollama-two"


def test_entrypoint_renders_named_provider_pools(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("VISION_LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("VISION_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("VISION_LLM_API_KEYS", "vision-one,vision-two")
    monkeypatch.setenv("HEAVY_CODING_LLM_MODEL", "coding-model")
    monkeypatch.setenv("HEAVY_CODING_LLM_BASE_URL", "https://coding.example/v1")
    monkeypatch.setenv("HEAVY_CODING_LLM_API_KEYS", "coding-one,coding-two")
    monkeypatch.setenv("OLLAMA_FALLBACK_LLM_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_FALLBACK_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_FALLBACK_LLM_API_KEYS", "ollama,ollama-two")
    rendered = entrypoint.render_llm_config()
    assert '[llm.vision]' in rendered and '"vision-one", "vision-two"' in rendered
    assert '[llm.heavy_coding]' in rendered and '"coding-one", "coding-two"' in rendered
    assert '[llm.fallback]' in rendered and '"ollama", "ollama-two"' in rendered
