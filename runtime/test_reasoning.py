from __future__ import annotations

import asyncio

from app.config import apply_llm_environment_overrides, config
from app.llm import LLM
from app.platform.reasoning import reasoning_enabled
from app.platform.specialists import SpecialistGateway, design_skill_context
from deploy import entrypoint


def test_deepseek_control_requires_explicit_configuration_or_enablement(monkeypatch):
    monkeypatch.delitem(config.llm, "reasoning", raising=False)
    monkeypatch.delenv("REASONING_LLM_ENABLED", raising=False)
    monkeypatch.delenv("REASONING_LLM_MODEL", raising=False)
    assert reasoning_enabled() is False

    monkeypatch.setenv("REASONING_LLM_MODEL", "deepseek-r1:7b")
    assert reasoning_enabled() is False  # an env label alone cannot stand in for a loaded role
    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    assert reasoning_enabled() is True
    monkeypatch.setenv("REASONING_LLM_ENABLED", "true")
    assert reasoning_enabled() is True


def test_deepseek_control_can_be_disabled_but_has_no_fallback(monkeypatch):
    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    monkeypatch.setenv("REASONING_LLM_MODEL", "deepseek-r1:7b")
    monkeypatch.setenv("REASONING_LLM_ENABLED", "false")
    assert reasoning_enabled() is False


def test_named_deepseek_role_does_not_fail_over_to_another_model(monkeypatch):
    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    assert LLM(config_name="reasoning")._fallback_config is None


def test_environment_model_roles_override_cloud_defaults_without_key_leakage():
    configured = {"default": {
        "model": "paid-default",
        "base_url": "https://provider.example/v1",
        "api_key": "cloud-secret",
        "api_type": "openai",
    }}
    env = {
        "LOCAL_LLM_MODEL": "qwen2.5-coder:7b",
        "LOCAL_LLM_BASE_URL": "http://ollama:11434/v1",
        "LOCAL_LLM_API_KEY": "ollama",
        "REASONING_LLM_MODEL": "deepseek-r1:7b",
        "RESEARCH_LLM_MODEL": "qwen2.5:3b",
        "CREATIVITY_LLM_MODEL": "llama3.2:3b",
    }
    roles = apply_llm_environment_overrides(configured, env)
    assert roles["default"]["model"] == "qwen2.5-coder:7b"
    assert roles["default"]["base_url"] == "http://ollama:11434/v1"
    assert roles["default"]["api_key"] == "ollama"
    assert roles["reasoning"]["model"] == "deepseek-r1:7b"
    assert roles["reasoning"]["base_url"] == "http://ollama:11434/v1"
    assert roles["reasoning"]["api_key"] == "ollama"
    assert roles["research"]["model"] == "qwen2.5:3b"
    assert roles["creativity"]["model"] == "llama3.2:3b"
    assert all(role["api_key"] != "cloud-secret" for role in roles.values())


def test_unconfigured_specialist_cannot_silently_use_default_model(monkeypatch):
    monkeypatch.delitem(config.llm, "research", raising=False)
    gateway = SpecialistGateway(config=config)
    try:
        gateway.llm("researcher")
    except RuntimeError as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("researcher must not silently fall back to the default model")


def test_malformed_specialist_json_emits_failure_not_completion(monkeypatch):
    events = []

    async def emit(kind, message, data):
        events.append(kind)

    class MalformedModel:
        model = "llama3.2:3b"

        async def ask(self, *args, **kwargs):
            return "This is not JSON."

    gateway = SpecialistGateway(config=config, emit=emit)
    monkeypatch.setattr(gateway, "llm", lambda role: MalformedModel())
    result = asyncio.run(gateway.ask_json("designer", instruction="Design the page"))
    assert result["parse_error"] == "specialist did not return JSON"
    assert events == ["specialist.started", "specialist.failed"]


def test_zip_design_references_are_normalized_for_local_design_handoffs():
    context = design_skill_context()
    assert "openmanus-design-playbook.md" in context
    playbook = next(iter(context.values()))
    assert "Responsive and accessible by default" in playbook
    assert "zero-cost default" in playbook


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
