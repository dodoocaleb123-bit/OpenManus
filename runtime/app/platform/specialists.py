"""Specialist-role routing and bounded fan-out helpers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class SpecialistRole:
    name: str
    purpose: str
    capabilities: tuple[str, ...]

ROLES = {
    "planner": SpecialistRole("planner", "Decompose complex requests and identify risks", ("planning", "risk")),
    "researcher": SpecialistRole("researcher", "Browse, compare, and cite external sources", ("research", "browser")),
    "coder": SpecialistRole("coder", "Implement changes and run project tests", ("coding", "workspace")),
    "vision": SpecialistRole("vision", "Interpret images and review visible UI", ("vision", "browser")),
    "verifier": SpecialistRole("verifier", "Check outputs independently and record evidence", ("verification", "testing")),
    "artifact": SpecialistRole("artifact", "Package documents, spreadsheets, and generated files", ("artifacts", "documents")),
}

def select_specialists(*, intent: str, complexity: str = "normal", has_images: bool = False, browser: bool = False, requires_artifacts: bool = False) -> list[dict[str, Any]]:
    names: list[str] = []
    if complexity == "heavy":
        names.append("planner")
    if intent == "research" or browser:
        names.append("researcher")
    if intent == "coding":
        names.append("coder")
    if has_images:
        names.append("vision")
    if requires_artifacts:
        names.append("artifact")
    if complexity in {"heavy", "normal"} or intent in {"coding", "research"}:
        names.append("verifier")
    unique = dict.fromkeys(names)
    return [{"name": name, "purpose": ROLES[name].purpose, "capabilities": list(ROLES[name].capabilities)} for name in unique]

async def bounded_gather(items: Iterable[T], worker: Callable[[T], Awaitable[Any]], *, limit: int = 3) -> list[Any]:
    """Run independent work with a strict concurrency bound and input order preserved."""
    values = list(items)
    semaphore = asyncio.Semaphore(max(1, min(int(limit), 16)))
    async def run(value: T) -> Any:
        async with semaphore:
            return await worker(value)
    return list(await asyncio.gather(*(run(value) for value in values)))
