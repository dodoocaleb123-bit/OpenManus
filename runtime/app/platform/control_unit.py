"""Registry-backed capability planning and deterministic safety validation."""
from __future__ import annotations
import re
from typing import Any
from app.platform.capability_registry import get_capability, registry_version, search_capabilities, validate_plan
from app.platform.handoffs import order_steps

class ControlUnit:
    """Builds executable plans; model output can choose IDs, platform owns execution."""
    def plan_from_capability_ids(self, request: str, capability_ids: list[int], *, attachment_ids: list[str] | None = None, browser_session_id: str | None = None, mode: str = "auto", excluded_capabilities: list[int] | None = None, controller: str = "deepseek", planner_status: str = "authoritative") -> dict[str, Any]:
        selected: list[int] = []
        excluded = {int(item) for item in (excluded_capabilities or [])}
        def add(cid: int) -> None:
            cid = int(cid)
            if cid in excluded or cid in selected: return
            get_capability(cid)
            selected.append(cid)
        def add_with_dependencies(cid: int) -> None:
            record = get_capability(int(cid))
            for dependency in record.get("dependencies", []): add_with_dependencies(int(dependency))
            add(int(cid))
        for cid in capability_ids: add_with_dependencies(int(cid))
        if not selected: add(167)
        step_for = {cid: f"step-{index}" for index, cid in enumerate(selected, 1)}
        steps: list[dict[str, Any]] = []
        for cid in selected:
            record = get_capability(cid)
            dependencies = [step_for[int(dep)] for dep in record.get("dependencies", []) if int(dep) in step_for]
            prior = [item for item in selected if item != cid]
            if record["handler"] in {"qwen_coder", "llama3.2_3b"}:
                dependencies.extend(step_for[item] for item in prior if get_capability(item)["handler"] in {"gemma3", "qwen2.5_3b"})
            steps.append({"step_id": step_for[cid], "capability_id": cid, "handler": record["handler"], "supporting_handlers": record["supporting_handlers"], "depends_on": list(dict.fromkeys(dependencies)), "input": {"user_request": request, "attachment_ids": attachment_ids or [], "dependency_outputs": {}}, "input_schema": record["input_schema"], "output_schema": record["output_schema"], "verification": record["verification"], "requires_user": record["requires_user"], "platform_only": record["platform_only"], "requires_confirmation": record["requires_confirmation"], "failure_policy": record["failure_policy"]})
        return validate_plan({"schema_version": "1.2.0", "registry_version": registry_version(), "request_type": mode, "selected_capabilities": selected, "excluded_capabilities": sorted(excluded), "steps": order_steps(steps), "attachment_ids": attachment_ids or [], "browser_session_id": browser_session_id, "controller": controller, "planner_status": planner_status, "authoritative": controller == "deepseek" and planner_status == "authoritative", "evidence_required": True})

    def plan(self, request: str, *, attachment_ids: list[str] | None = None, browser_session_id: str | None = None, mode: str = "auto") -> dict[str, Any]:
        """Compatibility fallback used only when DeepSeek is unavailable."""
        text = request.strip(); low = text.casefold(); selected: list[int] = []
        def add(cid: int):
            if cid not in selected: selected.append(cid)
        if attachment_ids or re.search(r"\b(image|screenshot|photo|diagram|picture)\b", low): add(131)
        if re.search(r"\b(research|search|sources?|internet|web|website|compare|investigate|url)\b|https?://", low): add(147)
        if re.search(r"\b(design|beautiful|ui|ux|landing page|visual|typography|palette|prototype)\b", low): add(135)
        if re.search(r"\b(build|create|implement|edit|fix|refactor|code|file|app|application|test|docker|deploy)\b", low): add(1)
        if re.search(r"\b(github|push|pull request|commit|publish|repository)\b", low): add(83)
        if re.search(r"\b(click|type|take over|captcha|mfa|password|payment|login)\b", low): add(231)
        if not selected: add(167)
        return self.plan_from_capability_ids(text, selected, attachment_ids=attachment_ids, browser_session_id=browser_session_id, mode=mode, controller="deepseek", planner_status="fallback")

    def relevant_registry(self, request: str, limit: int = 20) -> list[dict[str, Any]]:
        return search_capabilities(request, limit)
