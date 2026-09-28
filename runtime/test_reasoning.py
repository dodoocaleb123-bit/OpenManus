from __future__ import annotations

from app.platform.reasoning import reasoning_enabled, should_reason
from deploy import entrypoint


def test_reasoning_is_opt_in_and_selective(monkeypatch):
    monkeypatch.delenv("REASONING_LLM_ENABLED", raising=False)
    assert reasoning_enabled() is False
    assert should_reason("refactor the whole authentication architecture", complexity="heavy") is False

    monkeypatch.setenv("REASONING_LLM_ENABLED", "true")
    monkeypatch.setenv("REASONING_LLM_MODE", "complex")
    assert should_reason("say hello", complexity="normal") is False
    assert should_reason("refactor the whole authentication architecture", complexity="heavy") is True
    assert should_reason("delete the old database migration", complexity="normal") is True


def test_reasoning_mode_can_be_disabled(monkeypatch):
    monkeypatch.setenv("REASONING_LLM_ENABLED", "true")
    monkeypatch.setenv("REASONING_LLM_MODE", "off")
    assert should_reason("refactor the whole application", complexity="heavy") is False


def test_entrypoint_renders_reasoning_profile(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://host.docker.internal:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "ollama")
    monkeypatch.setenv("REASONING_LLM_MODEL", "deepseek-r1:7b")
    monkeypatch.setenv("REASONING_LLM_BASE_URL", "http://host.docker.internal:11434/v1")
    monkeypatch.setenv("REASONING_LLM_API_KEY_01", "ollama")
    rendered = entrypoint.render_llm_config()
    assert '[llm.reasoning]' in rendered
    assert 'model = "deepseek-r1:7b"' in rendered
    assert 'api_keys = ["ollama"]' in rendered
