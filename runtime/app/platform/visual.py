"""Optional visual verification for browser-based UI tasks."""
from __future__ import annotations

import os
from typing import Any


def visual_verification_enabled() -> bool:
    return os.environ.get("VISUAL_VERIFY_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def visual_prompt(request: str) -> str:
    return (
        "Inspect this browser screenshot as a visual QA reviewer. Do not edit files. "
        "Return concise JSON with keys: passed (boolean), concerns (array of strings), "
        "and observations (array of strings). Mention only visible UI issues such as overlap, "
        "missing content, unreadable contrast, broken layout, or obvious loading/error states. "
        f"User request: {request}"
    )
