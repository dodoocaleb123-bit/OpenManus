"""Descriptive OpenManus abilities exposed to the DeepSeek control unit.

This is intentionally human-readable. It is planning context, not an execution
registry; executable adapters remain private to the platform.
"""
from __future__ import annotations

import json


MANIFEST_VERSION = "1.0"


def build_ability_manifest(configured_models: list[dict] | None = None) -> str:
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "purpose": "Describe outcomes OpenManus can deliver and the best owner for each kind of work.",
        "models": {
            "deepseek": {
                "role": "authoritative control unit and final integrator",
                "can": ["understand the user's complete intent", "ask clarifying questions", "decompose outcomes into ordered steps", "choose models and tools", "review evidence and failures", "provide the final answer"],
                "cannot": ["directly edit files or claim tool results it did not receive"],
            },
            "qwen_coder": {
                "role": "engineering and execution specialist",
                "can": ["inspect and edit project files", "run terminal commands", "create applications", "debug code", "run tests", "start local previews", "use approved GitHub actions through the platform"],
            },
            "gemma3": {
                "role": "vision specialist",
                "can": ["analyze uploaded images and screenshots", "extract visible text", "identify colors, layout, and visual hierarchy", "compare visual references"],
            },
            "qwen2.5_3b": {
                "role": "research specialist",
                "can": ["inspect public web pages", "summarize sources", "extract facts with citations", "compare implementation options"],
            },
            "llama3.2_3b": {
                "role": "design and creative specialist",
                "can": ["propose UX and visual direction", "draft copy and concepts", "recommend layout, typography, and interaction patterns"],
            },
            "platform": {
                "role": "deterministic execution services",
                "can": ["manage files and project workspaces", "run local tests", "start private previews", "operate the shared browser", "record evidence", "perform approved GitHub operations"],
            },
            "user": {
                "role": "human-in-the-loop",
                "can": ["provide missing information", "approve risky actions", "complete browser or external-account steps"],
            },
        },
        "tools": [
            "filesystem", "terminal", "python", "browser", "image_analysis", "web_research",
            "preview_server", "test_runner", "git", "github", "ask_user", "evidence_store",
        ],
        "rules": [
            "Use the model that can best perform each step; use deterministic platform tools when sufficient.",
            "Pass only useful structured findings, files, evidence, and errors between steps.",
            "Do not expose private chain-of-thought.",
            "Pause for the user when an external action, approval, credential, or missing input is required.",
            "Every plan must include a final DeepSeek answer or clarification step.",
        ],
        "configured_models": configured_models or [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
