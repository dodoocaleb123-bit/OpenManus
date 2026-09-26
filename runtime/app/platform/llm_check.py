"""Detect an unusable LLM configuration before starting an agent run."""

from __future__ import annotations

from typing import Any

_PLACEHOLDER_KEYS = {"", "your_api_key", "sk-...", "changeme", "none", "null"}

SETUP_HINT = (
    "Set LLM_MODEL, LLM_BASE_URL and LLM_API_KEY in your Render service "
    "(Dashboard -> your service -> Environment), then redeploy. "
    "See DEPLOY.md for provider-specific values."
)


def llm_status() -> dict[str, Any]:
    """Summary of the active model configuration (never includes the key)."""
    try:
        from app.config import config
        from app.llm import model_supports_images
    except Exception as exc:  # config file unreadable
        return {"configured": False, "problem": f"Configuration could not be loaded: {exc}. {SETUP_HINT}"}

    try:
        settings = config.llm["default"]
    except Exception:
        return {"configured": False, "problem": f"No [llm] section in config. {SETUP_HINT}"}

    source = str(config._get_config_path()) if hasattr(config, "_get_config_path") else ""
    api_type = (getattr(settings, "api_type", "") or "").lower()
    key = (getattr(settings, "api_key", "") or "").strip()
    configured_keys = [key, *(getattr(settings, "api_keys", []) or [])]
    key_count = len(dict.fromkeys(item.strip() for item in configured_keys if item and item.strip()))
    problem = None
    if source.endswith("config.example.toml"):
        problem = f"No LLM configuration found (the example config is in use). {SETUP_HINT}"
    elif api_type != "aws" and key.lower() in _PLACEHOLDER_KEYS:
        problem = f"The LLM API key is missing or still a placeholder. {SETUP_HINT}"
    elif not getattr(settings, "model", ""):
        problem = f"No LLM model configured. {SETUP_HINT}"
    return {
        "configured": problem is None,
        "problem": problem,
        "model": getattr(settings, "model", None),
        "base_url": getattr(settings, "base_url", None),
        "api_type": api_type or "openai",
        "supports_images": model_supports_images(getattr(settings, "model", "") or ""),
        "api_key_count": key_count,
        "failover_configured": key_count > 1,
    }


def llm_problem() -> str | None:
    return llm_status().get("problem")
