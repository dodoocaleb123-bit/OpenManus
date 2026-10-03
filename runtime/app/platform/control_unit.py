"""Validation and dependency ordering for DeepSeek-authored workflows.

This module deliberately contains no task catalog or numbered capability registry.
DeepSeek chooses model handlers, objectives, tool use, and dependencies from
human-readable model and tool descriptions; the platform validates that the
chosen adapters are real and that the workflow is executable.
"""
from __future__ import annotations

import re
from typing import Any

from app.platform.handoffs import order_steps

WORKFLOW_HANDLERS = frozenset({"qwen_coder", "gemma3", "qwen2.5_3b", "llama3.2_3b", "user"})
WORKFLOW_TOOLS = frozenset({
    "bash", "python_execute", "str_replace_editor", "web_search",
    "platform_browser", "platform_git", "ask_human", "terminate",
})
GITHUB_ACTIONS = frozenset({
    "connect", "status", "diff", "log", "init", "create_branch", "checkout",
    "commit", "push", "create_pull_request", "publish_repository",
})
_STEP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ControlUnit:
    """Validate a structured workflow without deciding what the user meant."""

    def build_workflow(
        self,
        request: str,
        steps: list[dict[str, Any]],
        *,
        attachment_ids: list[str] | None = None,
        browser_session_id: str | None = None,
        controller: str = "deepseek",
        planner_status: str = "authoritative",
    ) -> dict[str, Any]:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(steps):
            if not isinstance(raw, dict):
                raise ValueError(f"Workflow step {index + 1} must be an object")
            step_id = str(raw.get("step_id") or "").strip()
            handler = str(raw.get("handler") or "").strip()
            objective = str(raw.get("objective") or "").strip()
            if not _STEP_ID.fullmatch(step_id):
                raise ValueError(f"Workflow step {index + 1} needs a valid step_id")
            if handler not in WORKFLOW_HANDLERS:
                raise ValueError(f"DeepSeek selected an unavailable workflow handler: {handler or '(empty)'}")
            if not objective:
                raise ValueError(f"Workflow step {step_id} needs a clear objective")
            depends_on = raw.get("depends_on", [])
            if not isinstance(depends_on, list) or any(not isinstance(dep, str) for dep in depends_on):
                raise ValueError(f"Workflow step {step_id} depends_on must be an array of step IDs")
            tools = raw.get("tools", [])
            if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
                raise ValueError(f"Workflow step {step_id} tools must be an array of tool names")
            unknown_tools = sorted(set(tools) - WORKFLOW_TOOLS)
            if unknown_tools:
                raise ValueError(f"Workflow step {step_id} selected unavailable tool(s): " + ", ".join(unknown_tools))
            if handler == "qwen_coder" and not tools:
                raise ValueError(f"Coding step {step_id} must explicitly select the OpenManus tools it needs")
            if handler == "qwen2.5_3b" and not {"web_search", "platform_browser"}.intersection(tools):
                raise ValueError(f"Research step {step_id} must explicitly select web_search or platform_browser")
            git_actions = raw.get("git_actions", [])
            if not isinstance(git_actions, list) or any(not isinstance(action, str) for action in git_actions):
                raise ValueError(f"Workflow step {step_id} git_actions must be an array")
            unknown_actions = sorted(set(git_actions) - GITHUB_ACTIONS)
            if unknown_actions:
                raise ValueError(f"Workflow step {step_id} selected unknown GitHub action(s): " + ", ".join(unknown_actions))
            if git_actions and "platform_git" not in tools:
                raise ValueError(f"Workflow step {step_id} requested GitHub actions without selecting platform_git")
            requires_confirmation = raw.get("requires_confirmation", False)
            if not isinstance(requires_confirmation, bool):
                raise ValueError(f"Workflow step {step_id} requires_confirmation must be a boolean")
            normalized.append({
                "step_id": step_id,
                "handler": handler,
                "objective": objective[:2000],
                "depends_on": list(dict.fromkeys(depends_on)),
                "tools": list(dict.fromkeys(tools)),
                "git_actions": list(dict.fromkeys(git_actions)),
                "requires_confirmation": requires_confirmation,
            })
        ordered = order_steps(normalized)
        if sum(step["handler"] == "user" for step in ordered) > 3:
            raise ValueError("A workflow may request at most three separate user actions")
        ids = {step["step_id"] for step in ordered}
        if any(step["step_id"] in step["depends_on"] for step in ordered):
            raise ValueError("A workflow step cannot depend on itself")
        if any(dep not in ids for step in ordered for dep in step["depends_on"]):
            raise ValueError("Workflow contains a dependency on an unknown step")
        return {
            "workflow_version": "2.0",
            "request_type": "deepseek_authored_workflow",
            "steps": ordered,
            "attachment_ids": list(dict.fromkeys(attachment_ids or [])),
            "browser_session_id": browser_session_id,
            "controller": controller,
            "planner_status": planner_status,
            "authoritative": controller == "deepseek" and planner_status == "authoritative",
            "evidence_required": True,
            "user_request": request,
        }
