"""Deterministic request classification used before model planning/execution."""
from __future__ import annotations

import re
from typing import Any


_PATTERNS = {
    "research": r"\b(research|search|browse|sources?|compare|summari[sz]e|internet|web page|website)\b",
    "image": r"\b(image|photo|picture|screenshot|diagram|handwritten|visual)\b",
    "data": r"\b(csv|excel|spreadsheet|dataset|data analysis|chart|plot|statistics)\b",
    "document": r"\b(pdf|docx?|report|presentation|slides?|document)\b",
    "coding": r"\b(code|coding|implement|build|create|edit|fix|refactor|debug|test|deploy|repository|repo|file)\b",
    "conversation": r"\b(hello|hi|thanks|what|why|how|explain|tell me)\b",
}

_COMPLEX = re.compile(
    r"\b(architect|architecture|migrat|refactor|restructure|redesign|production|"
    r"authentication|authorization|database|schema|deploy|security|multiple|entire|"
    r"whole|large|complex|integrat|debug|diagnos|plan|review|several|multi[- ]file|"
    r"full application|from scratch)\b",
    re.IGNORECASE,
)
_RISK = re.compile(
    r"\b(delete|remove|drop|destroy|overwrite|credential|secret|permission|publish|"
    r"push|force|billing|migration|breaking|irreversible|production)\b",
    re.IGNORECASE,
)


def classify_request(prompt: str, *, browser: bool = False, has_images: bool = False) -> dict[str, Any]:
    """Return bounded, deterministic routing metadata; never calls an LLM."""
    text = (prompt or "").strip()
    scores = {name: len(re.findall(pattern, text, re.IGNORECASE)) for name, pattern in _PATTERNS.items()}
    if has_images:
        scores["image"] += 3
    if browser:
        scores["research"] += 2
    intent = max(scores, key=scores.get) if any(scores.values()) else "conversation"
    if has_images:
        intent = "image"
    elif browser and intent in {"conversation", "research"}:
        intent = "research"
    complexity = "heavy" if _COMPLEX.search(text) or len(text) > 900 or browser else "normal"
    if re.search(r"\b(create|make)\s+(a\s+)?file\b", text, re.IGNORECASE) and len(text) < 500:
        complexity = "simple"
    return {
        "intent": intent,
        "complexity": complexity,
        "risk": "high" if _RISK.search(text) else "normal",
        "scores": scores,
        "browser": bool(browser),
        "has_images": bool(has_images),
        "requires_reasoning": complexity == "heavy" or bool(_RISK.search(text)),
    }
