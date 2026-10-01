"""DeepSeek control-unit planning independent from tool execution."""
from __future__ import annotations

import re
from typing import Any

from app.platform.capability_registry import capabilities, registry_version, search_capabilities
from app.platform.handoffs import order_steps


class ControlUnit:
    """Builds plans; platform services remain authoritative executors."""

    def plan(self, request: str, *, attachment_ids: list[str] | None = None, browser_session_id: str | None = None, mode: str = "auto") -> dict[str, Any]:
        text = request.strip()
        low = text.casefold()
        selected: list[dict[str, Any]] = []
        def add(cid: int):
            record = next(r for r in capabilities() if r["id"] == cid)
            if record not in selected: selected.append(record)
        has_image = bool(attachment_ids) or bool(re.search(r"\b(image|screenshot|photo|diagram|picture)\b", low))
        research = bool(re.search(r"\b(research|search|sources?|internet|web|website|compare|investigate|url)\b|https?://", low))
        design = bool(re.search(r"\b(design|beautiful|ui|ux|landing page|visual|typography|palette|prototype)\b", low))
        coding = bool(re.search(r"\b(build|create|implement|edit|fix|refactor|code|file|app|application|test|docker|deploy)\b", low))
        github = bool(re.search(r"\b(github|push|pull request|commit|publish|repository)\b", low))
        user_action = bool(re.search(r"\b(click|type|take over|captcha|mfa|password|payment|login)\b", low))
        if has_image: add(131)
        if research: add(147)
        if design: add(135)
        if coding: add(1)
        if github: add(83)
        if user_action: add(231)
        if not selected: add(167)
        steps=[]
        for n, record in enumerate(selected, 1):
            depends = [f"step-{i}" for i, prior in enumerate(selected[:n-1], 1) if record["handler"] in {"qwen_coder", "llama3.2_3b", "deepseek"} and prior["handler"] in {"gemma3", "qwen2.5_3b", "llama3.2_3b"}]
            steps.append({"step_id":f"step-{n}","capability_id":record["id"],"handler":record["handler"],"depends_on":depends,"input":{"attachment_ids":attachment_ids or [],"request":text},"verification":record["verification"],"requires_user":record["requires_user"],"platform_only":record["platform_only"]})
        ordered=order_steps(steps)
        return {"schema_version":"1.0.0","registry_version":registry_version(),"request_type":mode,"selected_capabilities":[r["id"] for r in selected],"excluded_capabilities":[],"steps":ordered,"attachment_ids":attachment_ids or [],"browser_session_id":browser_session_id,"controller":"deepseek","evidence_required":True}

    def relevant_registry(self, request: str, limit: int = 20) -> list[dict[str, Any]]:
        return search_capabilities(request, limit)
