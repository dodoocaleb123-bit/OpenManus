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
    """Enable the DeepSeek control unit by default when configured.

    Deployments can explicitly disable it with ``REASONING_LLM_ENABLED=false``
    for constrained hardware or diagnostics.
    """
    value = os.environ.get("REASONING_LLM_ENABLED", "").strip().casefold()
    if value in {"0", "false", "no", "off"}:
        return False
    return value in {"1", "true", "yes", "on"} or bool(os.environ.get("REASONING_LLM_MODEL"))


def should_reason(prompt: str, *, complexity: str = "normal") -> bool:
    """Use reasoning only for complex, risky, or explicitly planning requests."""
    if not reasoning_enabled():
        return False
    mode = os.environ.get("REASONING_LLM_MODE", "always").strip().casefold()
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


async def make_authoritative_plan(
    llm: LLM,
    *,
    prompt: str,
    attachment_ids: list[str] | None = None,
    browser_session_id: str | None = None,
    mode: str = "auto",
) -> dict[str, Any]:
    """Ask DeepSeek to select registry capabilities, then validate on-platform."""
    from app.platform.capability_registry import capabilities
    from app.platform.control_unit import ControlUnit

    catalog = [
        {"id": r["id"], "name": r["name"], "category": r["category"], "handler": r["handler"],
         "dependencies": r.get("dependencies", []), "requires_user": r["requires_user"],
         "requires_confirmation": r["requires_confirmation"]}
        for r in capabilities()
    ]
    response = await llm.ask(
        [{"role": "user", "content": (
            "You are the authoritative OpenManus control unit. Select the exact capability IDs needed "
            "for the request from the registry. Do not invent IDs, handlers, tools, dependencies, or "
            "completion claims. The platform will execute only your validated selections. Return ONLY JSON "
            "with keys: summary, intent, capability_ids (integer array), excluded_capabilities (integer array), "
            "requires_confirmation (boolean), rationale (string array). Prefer the smallest complete plan.\n\n"
            f"Registry:\n{json.dumps(catalog, ensure_ascii=False)}\n\nRequest: {prompt}"
        )}],
        stream=False,
        temperature=0.0,
        max_tokens=max(512, int(os.environ.get("REASONING_LLM_MAX_TOKENS", "1800"))),
    )
    value = _parse_json(response)
    if value.get("parse_error"):
        raise ValueError(value["parse_error"])
    ids = value.get("capability_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, int) or isinstance(item, bool) for item in ids):
        raise ValueError("DeepSeek authoritative plan must contain integer capability_ids")
    plan = ControlUnit().plan_from_capability_ids(
        prompt, ids, attachment_ids=attachment_ids, browser_session_id=browser_session_id,
        mode=mode, excluded_capabilities=value.get("excluded_capabilities") or [],
        controller="deepseek", planner_status="authoritative",
    )
    plan["deepseek"] = {
        "model": llm.model, "summary": str(value.get("summary") or "")[:2000],
        "intent": str(value.get("intent") or "unknown")[:120],
        "rationale": _string_list(value.get("rationale"), limit=20), "raw_response": response[:12000],
    }
    plan["requires_confirmation"] = bool(value.get("requires_confirmation", False)) or any(
        step.get("requires_confirmation") for step in plan["steps"]
    )
    return plan


async def make_plan(llm: LLM, *, prompt: str, conversation: str, workspace: str) -> dict[str, Any]:
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
                    "The project title is UI metadata only. Do not infer the task topic, domain, or intent from it.\n"
                    f"Workspace: {workspace}\n"
                    f"Relevant prior task context (only if explicitly referenced):\n{conversation[-12000:]}\n\n"
                    f"CURRENT USER REQUEST (authoritative):\n{prompt}"
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
