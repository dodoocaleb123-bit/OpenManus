"""Registry-backed validation of DeepSeek-selected capability plans."""
from __future__ import annotations
from typing import Any
from app.platform.capability_registry import get_capability, registry_version, validate_plan
from app.platform.handoffs import order_steps

class ControlUnit:
    """Builds executable plans; model output can choose IDs, platform owns execution."""
    def plan_from_capability_ids(self, request: str, capability_ids: list[int], *, attachment_ids: list[str] | None = None, browser_session_id: str | None = None, excluded_capabilities: list[int] | None = None, controller: str = "deepseek", planner_status: str = "authoritative") -> dict[str, Any]:
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
        if not selected:
            raise ValueError("DeepSeek must explicitly select at least one capability")
        step_for = {cid: f"step-{index}" for index, cid in enumerate(selected, 1)}
        steps: list[dict[str, Any]] = []
        for position, cid in enumerate(selected):
            record = get_capability(cid)
            dependencies = [step_for[int(dep)] for dep in record.get("dependencies", []) if int(dep) in step_for]
            prior = selected[:position]
            if record["handler"] == "qwen_coder":
                # Qwen must receive every earlier selected user/specialist
                # handoff. Later user actions remain true post-coding pauses.
                dependencies.extend(
                    step_for[item] for item in prior
                    if (get_capability(item)["handler"] in {"qwen2.5_3b", "llama3.2_3b", "user"})
                    or (get_capability(item)["handler"] == "gemma3" and item not in {132, 133})
                )
            elif record["handler"] == "llama3.2_3b":
                dependencies.extend(
                    step_for[item] for item in prior
                    if (get_capability(item)["handler"] == "qwen2.5_3b")
                    or (get_capability(item)["handler"] == "gemma3" and item not in {132, 133})
                )
            elif record["handler"] == "user":
                # A user action placed after a specialist is not merely a
                # preflight gate: it must pause only after earlier work ends.
                dependencies.extend(
                    step_for[item] for item in prior
                    if get_capability(item)["handler"] in {"qwen_coder", "qwen2.5_3b", "llama3.2_3b", "gemma3"}
                )
            steps.append({"step_id": step_for[cid], "capability_id": cid, "handler": record["handler"], "supporting_handlers": record["supporting_handlers"], "tools": record["tools"], "depends_on": list(dict.fromkeys(dependencies)), "input": {"user_request": request, "attachment_ids": attachment_ids or [], "dependency_outputs": {}}, "input_schema": record["input_schema"], "output_schema": record["output_schema"], "verification": record["verification"], "requires_user": record["requires_user"], "platform_only": record["platform_only"], "requires_confirmation": record["requires_confirmation"], "failure_policy": record["failure_policy"]})
        return validate_plan({"schema_version": "1.2.0", "registry_version": registry_version(), "request_type": "deepseek_selected_workflow", "selected_capabilities": selected, "excluded_capabilities": sorted(excluded), "steps": order_steps(steps), "attachment_ids": attachment_ids or [], "browser_session_id": browser_session_id, "controller": controller, "planner_status": planner_status, "authoritative": controller == "deepseek" and planner_status == "authoritative", "evidence_required": True})
