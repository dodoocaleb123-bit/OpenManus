"""Typed model-role metadata and safe environment resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    role: str
    config_name: str
    env_model: str
    default_model: str
    capability: str
    supports: tuple[str, ...]


PROFILES = (
    ModelProfile("qwen_coder", "heavy_coding", "LOCAL_LLM_MODEL", "qwen2.5-coder:7b", "coding", ("coding", "tools", "workspace")),
    ModelProfile("gemma3", "vision", "VISION_LLM_MODEL", "gemma3:4b", "vision", ("vision", "image_analysis", "visual_review")),
    ModelProfile("deepseek", "reasoning", "REASONING_LLM_MODEL", "deepseek-r1:7b", "reasoning", ("reasoning", "planning", "orchestration")),
    ModelProfile("qwen2.5_3b", "research", "RESEARCH_LLM_MODEL", "qwen2.5:3b", "research", ("research", "sources", "browser")),
    ModelProfile("llama3.2_3b", "creativity", "CREATIVITY_LLM_MODEL", "llama3.2:3b", "creativity", ("creativity", "design", "content")),
)


def profiles() -> tuple[ModelProfile, ...]:
    return PROFILES


def status_from_env() -> list[dict[str, Any]]:
    return [{
        "role": profile.role,
        "config_name": profile.config_name,
        "model": os.getenv(profile.env_model, profile.default_model),
        "configured": bool(os.getenv(profile.env_model)),
        "capability": profile.capability,
        "supports": list(profile.supports),
        "status": "configured" if os.getenv(profile.env_model) else "default",
    } for profile in PROFILES]
