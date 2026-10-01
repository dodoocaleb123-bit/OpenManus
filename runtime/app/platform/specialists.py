"""Specialist model routing and bounded multi-model handoffs.

The gateway is deliberately tool-free: DeepSeek or the platform orchestrator
selects a specialist, while platform tools remain the only way to mutate files,
run commands, browse, or perform protected external actions.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, TypeVar

from app.llm import LLM

T = TypeVar("T")


@dataclass(frozen=True)
class SpecialistRole:
    name: str
    purpose: str
    capabilities: tuple[str, ...]
    model_config_name: str


ROLES = {
    "planner": SpecialistRole("planner", "Reasoning, decomposition, and risk analysis", ("planning", "risk"), "reasoning"),
    "researcher": SpecialistRole("researcher", "Source-grounded internet research and synthesis", ("research", "sources"), "research"),
    "coder": SpecialistRole("coder", "Software engineering, testing, and repository work", ("coding", "workspace", "github"), "heavy_coding"),
    "vision": SpecialistRole("vision", "Image, screenshot, and visual-result analysis", ("vision", "visual_review"), "vision"),
    "designer": SpecialistRole("designer", "Creative direction, content, and beautiful UI design", ("creativity", "design", "content"), "creativity"),
    "verifier": SpecialistRole("verifier", "Independent evidence and quality checks", ("verification", "testing"), "reasoning"),
    "artifact": SpecialistRole("artifact", "Documents, spreadsheets, and generated-file packaging", ("artifacts", "documents"), "default"),
}


def is_design_request(prompt: str) -> bool:
    return bool(re.search(
        r"\b(design|beautiful|beauty|visual identity|ui|ux|interface|landing page|dashboard|mobile app|webpage|screen|layout|typography|color palette|wireframe|prototype)\b",
        prompt or "", re.IGNORECASE,
    ))


def is_research_request(prompt: str) -> bool:
    return bool(re.search(
        r"\b(research|deep research|investigate|sources?|cite|compare|literature|find out|search the internet|search online|web research)\b",
        prompt or "", re.IGNORECASE,
    ))


def specialist_configured(role: str, config: Any) -> bool:
    config_name = ROLES.get(role, ROLES["coder"]).model_config_name
    return config_name in getattr(config, "llm", {})


def select_specialists(*, intent: str, complexity: str = "normal", has_images: bool = False, browser: bool = False, requires_artifacts: bool = False, design: bool = False, research: bool = False) -> list[dict[str, Any]]:
    names: list[str] = []
    if complexity == "heavy":
        names.append("planner")
    if research or intent == "research" or browser:
        names.append("researcher")
    if design:
        names.append("designer")
    if intent in {"coding", "engineering"} or design:
        names.append("coder")
    if has_images:
        names.append("vision")
    if requires_artifacts:
        names.append("artifact")
    if complexity in {"heavy", "normal"} or intent in {"coding", "research", "engineering"}:
        names.append("verifier")
    unique = dict.fromkeys(names)
    return [
        {"name": name, "purpose": ROLES[name].purpose, "capabilities": list(ROLES[name].capabilities), "model_config": ROLES[name].model_config_name}
        for name in unique
    ]


def _parse_json(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return {"raw": text, "parse_error": "specialist did not return JSON"}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": "specialist returned malformed JSON"}
    return value if isinstance(value, dict) else {"raw": text, "parse_error": "specialist response was not an object"}


def _clip(value: Any, limit: int = 12000) -> str:
    return str(value or "")[:limit]


class SpecialistGateway:
    """Bounded, auditable calls to configured specialist models."""

    def __init__(self, config: Any, emit: Callable[[str, str, dict], Awaitable[None]] | None = None):
        self.config = config
        self.emit = emit

    async def _event(self, kind: str, message: str, data: dict[str, Any]) -> None:
        if self.emit:
            await self.emit(kind, message, data)

    def llm(self, role: str) -> LLM:
        config_name = ROLES[role].model_config_name
        if config_name not in self.config.llm:
            raise RuntimeError(f"Specialist model is not configured: {config_name}")
        return LLM(config_name=config_name)

    async def ask_json(self, role: str, *, instruction: str, context: dict[str, Any] | None = None, max_tokens: int = 1800) -> dict[str, Any]:
        llm = self.llm(role)
        await self._event("specialist.started", f"{role} specialist started", {"role": role, "model": llm.model})
        response = await llm.ask(
            [{"role": "user", "content": instruction + "\n\nCONTEXT:\n" + json.dumps(context or {}, ensure_ascii=False, default=str)[:30000]}],
            stream=False,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        parsed = _parse_json(response)
        result = {"role": role, "model": llm.model, "raw_response": _clip(response), **parsed}
        await self._event("specialist.completed", f"{role} specialist completed", {"role": role, "model": llm.model, "result": result})
        return result

    async def analyze_images(self, *, prompt: str, image_urls: list[str], context: dict[str, Any] | None = None) -> dict[str, Any]:
        llm = self.llm("vision")
        await self._event("specialist.started", "vision specialist started", {"role": "vision", "model": llm.model, "image_count": len(image_urls)})
        response = await llm.ask_with_images(
            [{"role": "user", "content": prompt}], image_urls, stream=False, temperature=0.1,
        )
        result = {"role": "vision", "model": llm.model, "raw_response": _clip(response), "analysis": _clip(response), "context": context or {}}
        await self._event("specialist.completed", "vision specialist completed", {"role": "vision", "model": llm.model, "result": result})
        return result


async def bounded_gather(items: Iterable[T], worker: Callable[[T], Awaitable[Any]], *, limit: int = 3) -> list[Any]:
    """Run independent work with a strict concurrency bound and input order preserved."""
    values = list(items)
    semaphore = asyncio.Semaphore(max(1, min(int(limit), 16)))

    async def run(value: T) -> Any:
        async with semaphore:
            return await worker(value)

    return list(await asyncio.gather(*(run(value) for value in values)))
