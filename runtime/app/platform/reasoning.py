"""Selective reasoning-model planning and review for platform tasks."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from app.llm import LLM


_COMPLEX_RE = re.compile(
    r"\b(architect|architecture|migrat|refactor|restructure|redesign|production|"
    r"authentication|authorization|database|schema|deploy|security|multiple|entire|"
    r"whole|large|complex|integrat|debug|diagnos|analy[sz]e|plan|review|"
    r"several|multi[- ]file|full application)\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"\b(delete|remove|drop|destroy|overwrite|credential|secret|permission|"
    r"publish|push|force|billing|migration|breaking|irreversible)\b",
    re.IGNORECASE,
)


def reasoning_enabled() -> bool:
    """Require explicit opt-in so adding the profile never slows all tasks."""
    return os.environ.get("REASONING_LLM_ENABLED", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }


def should_reason(prompt: str, *, complexity: str = "normal") -> bool:
    """Use reasoning only for complex, risky, or explicitly planning requests."""
    if not reasoning_enabled():
        return False
    mode = os.environ.get("REASONING_LLM_MODE", "complex").strip().casefold()
    if mode in {"off", "disabled", "never"}:
        return False
    if mode in {"always", "all"}:
        return True
    return complexity == "heavy" or bool(_COMPLEX_RE.search(prompt) or _RISK_RE.search(prompt))


def _parse_json(text: str) -> dict[str, Any]:
    """Parse strict or fenced JSON while retaining the raw response on failure."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return {"raw": text, "parse_error": "reasoning model did not return JSON"}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": "reasoning model returned malformed JSON"}
    return value if isinstance(value, dict) else {"raw": text, "parse_error": "reasoning response was not an object"}


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:500] for item in value if str(item).strip()][:limit]


def _normalise_plan(value: dict[str, Any], model: str, raw: str) -> dict[str, Any]:
    """Return a bounded advisory plan with a stable schema."""
    return {
        "summary": str(value.get("summary") or "").strip()[:2000],
        "intent": str(value.get("intent") or "unknown").strip()[:120],
        "steps": _string_list(value.get("steps")),
        "affected_areas": _string_list(value.get("affected_areas")),
        "risks": _string_list(value.get("risks")),
        "tests": _string_list(value.get("tests")),
        "requires_confirmation": bool(value.get("requires_confirmation", False)),
        "model": model,
        "raw_response": raw[:12000],
        "valid": bool(value.get("summary")) and bool(_string_list(value.get("steps"))),
    }


def _normalise_review(value: dict[str, Any], model: str, raw: str) -> dict[str, Any]:
    return {
        "passed": bool(value.get("passed", False)),
        "concerns": _string_list(value.get("concerns")),
        "recommended_follow_up": str(value.get("recommended_follow_up") or "").strip()[:2000],
        "model": model,
        "raw_response": raw[:12000],
        "valid": isinstance(value.get("passed"), bool),
    }


async def make_plan(llm: LLM, *, prompt: str, conversation: str, project_name: str, workspace: str) -> dict[str, Any]:
    """Ask the reasoning model for an advisory plan; never execute its output."""
    response = await llm.ask(
        [
            {
                "role": "user",
                "content": (
                    "Create a concise implementation plan for the OpenManus executor. "
                    "Do not edit files, call tools, or invent repository details. "
                    "Return ONLY valid JSON with keys: summary (string), intent (string), "
                    "steps (array of strings), affected_areas (array of strings), risks "
                    "(array of strings), tests (array of strings), requires_confirmation (boolean).\n\n"
                    f"Project: {project_name}\nWorkspace: {workspace}\n"
                    f"Recent conversation:\n{conversation[-12000:]}\n\nUser request:\n{prompt}"
                ),
            }
        ],
        stream=False,
        temperature=0.0,
        max_tokens=max(256, int(os.environ.get("REASONING_LLM_MAX_TOKENS", "1200"))),
    )
    return _normalise_plan(_parse_json(response), llm.model, response)


async def review_result(
    llm: LLM,
    *,
    prompt: str,
    summary: str,
    validation: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Ask for an advisory post-validation review; deterministic checks remain authoritative."""
    response = await llm.ask(
        [
            {
                "role": "user",
                "content": (
                    "Review this completed OpenManus task. Do not call tools and do not change files. "
                    "Return ONLY valid JSON with keys: passed (boolean), concerns (array of strings), "
                    "recommended_follow_up (string). Treat deterministic validation as authoritative; "
                    "flag only clear omissions or contradictions.\n\n"
                    f"Request: {prompt}\nSummary: {summary}\n"
                    f"Validation: {json.dumps(validation or {}, default=str)}\n"
                    f"Evidence: {json.dumps(evidence, default=str)[:12000]}"
                ),
            }
        ],
        stream=False,
        temperature=0.0,
        max_tokens=max(256, int(os.environ.get("REASONING_LLM_MAX_TOKENS", "1200"))),
    )
    return _normalise_review(_parse_json(response), llm.model, response)
