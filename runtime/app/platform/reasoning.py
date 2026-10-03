"""Mandatory DeepSeek control-unit planning and review for platform requests."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import BadRequestError

from app.llm import LLM, strip_think_tags


OPENMANUS_IDENTITY_INSTRUCTION = (
    "OpenManus is the name of the assistant speaking to the user; DeepSeek is the underlying reasoning model, not the assistant's name. "
    "When asked your name, identify yourself as OpenManus. Do not answer with the user's name, claim to be the human creator, or confuse the assistant with Qwen, Gemma, or another model. "
    "For a simple greeting or question about OpenManus's name, answer directly and select no tools or specialist workflow steps."
)


def reasoning_enabled() -> bool:
    """Report whether the mandatory DeepSeek control model is configured.

    ``REASONING_LLM_ENABLED=false`` does not enable a fallback classifier; it
    makes new user requests unavailable until the local controller is enabled.
    """
    value = os.environ.get("REASONING_LLM_ENABLED", "").strip().casefold()
    if value in {"0", "false", "no", "off"}:
        return False
    from app.config import config

    return "reasoning" in config.llm and bool(config.llm["reasoning"].model)


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


async def _ask_controller_json(
    llm: LLM,
    prompt: str,
    system: str,
    *,
    required_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Ask the *same* DeepSeek model for a small structured control decision.

    Ollama's OpenAI-compatible endpoint supports JSON response_format. A short,
    bounded retry on malformed/think-only output is model repair, never an
    alternate classifier or a substitute model. Never log raw private reasoning.
    """
    endpoint = str(getattr(llm, "base_url", "") or "")
    json_mode = {"type": "json_object"} if ":11434" in endpoint else None
    token_budget = max(2048, int(os.environ.get("OPENMANUS_CONTROLLER_MAX_TOKENS", "3072")))
    schema_errors: list[str] = []
    previous_value: dict[str, Any] | None = None
    for attempt in range(2):
        attempt_prompt = prompt
        attempt_system = system + ("\nReturn ONLY a short JSON object now; omit any thinking text." if attempt else "")
        if attempt and schema_errors:
            attempt_system += "\nCorrect the schema errors and return the complete corrected JSON object."
            attempt_prompt += (
                "\n\nPrevious JSON response:\n"
                + json.dumps(previous_value or {}, ensure_ascii=False)[:3000]
                + "\nSchema errors to correct: "
                + "; ".join(schema_errors)
            )
        try:
            response = await llm.ask(
                [{"role": "user", "content": attempt_prompt}],
                system_msgs=[{"role": "system", "content": attempt_system}],
                stream=False,
                temperature=0.0,
                max_tokens=token_budget if attempt == 0 else max(4096, token_budget),
                response_format=json_mode,
            )
        except BadRequestError as exc:
            if json_mode is not None and ("response_format" in str(exc).lower() or "json mode" in str(exc).lower()):
                # Older local Ollama endpoints may not implement the otherwise
                # supported OpenAI-compatible response_format field.
                json_mode = None
                if attempt == 0:
                    continue
            raise
        except ValueError as exc:
            # Ollama may return no visible content when a 7B reasoning model
            # exhausts its budget on private thinking before producing JSON.
            if attempt == 0 and "Empty or invalid response" in str(exc):
                continue
            raise
        visible = strip_think_tags(response)
        value = _parse_json(visible)
        if value.get("parse_error") or not visible:
            schema_errors = []
            previous_value = None
            continue
        schema_errors = []
        for field in required_fields:
            if field not in value:
                schema_errors.append(f"missing required field '{field}'")
            elif field == "summary" and not str(value.get(field) or "").strip():
                schema_errors.append("required field 'summary' must be a non-empty string")
            elif field == "needs_workspace_context" and not isinstance(value.get(field), bool):
                schema_errors.append("required field 'needs_workspace_context' must be a boolean")
            elif field == "workflow" and not isinstance(value.get(field), list):
                schema_errors.append("required field 'workflow' must be an array")
        if not schema_errors:
            return value
        previous_value = value
    raise ValueError("DeepSeek returned no usable JSON control decision after a same-model retry; check its response token budget and Ollama logs")


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
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask DeepSeek once for an ordered workflow from descriptive model/tool context.

    No numbered task catalog is presented to the model. DeepSeek chooses which
    available model handlers and tools to use, what each step should accomplish,
    and how the steps depend on one another. Platform validation checks only
    that the requested adapters exist and that the resulting graph is safe to run.
    """
    from app.platform.control_unit import ControlUnit

    context_value = dict(context or {})
    role_descriptions = {
        "deepseek": "Primary reasoning and control model. Interprets every user message, plans the workflow, handles direct answers, and synthesizes the final response after any delegated work. It does not execute workspace/browser/GitHub tools itself.",
        "qwen_coder": "Software engineering and execution model. Can inspect and modify the project workspace, run project commands/tests, use the browser preview, and invoke explicitly available OpenManus tools.",
        "gemma3": "Local vision model for interpreting user-provided images/screenshots and, when asked, inspecting rendered UI screenshots. It is not a general tool executor.",
        "qwen2.5_3b": "Local research specialist. Can form search queries, synthesize retrieved web content, compare sources, and return source-grounded findings.",
        "llama3.2_3b": "Local creative/design specialist. Can create practical visual, UX, content, and design guidance for the execution model.",
        "user": "Human action or missing-information step. Use only when work genuinely needs the user to act or provide information; state the exact action needed and resume after the user replies.",
    }
    tool_descriptions = {
        "bash": "Project-scoped terminal for shell commands, builds, tests, and local servers; not the user's host OS.",
        "python_execute": "Run Python scripts in the project workspace.",
        "str_replace_editor": "Read and edit project files in the workspace.",
        "web_search": "Search the web and retrieve source text when external research is needed and network policy allows it.",
        "platform_browser": "Shared browser session for user-visible navigation, research, and preview testing; session takeover can require the user's input.",
        "platform_git": "Scoped project Git/GitHub adapter: connect/status/diff/log/init/branch/checkout/commit/push/pull-request/publish. Pushes, publishing, and other protected remote actions still require explicit user request or approval.",
        "ask_human": "Pause and ask the user a specific question or request a required action; never request passwords, payment card details, or secrets in chat.",
        "terminate": "End the execution agent after its assigned work is complete.",
    }
    configured = context_value.get("configured_models", [])
    available_roles = set(context_value.get("available_model_roles", []))
    handler_roles = {
        "qwen_coder": {"heavy_coding", "default"},
        "gemma3": {"vision"},
        "deepseek": {"reasoning"},
        "qwen2.5_3b": {"research"},
        "llama3.2_3b": {"creativity"},
    }
    default_tools = list(tool_descriptions)
    available_tools = context_value.get("available_tools")
    if not isinstance(available_tools, list):
        available_tools = default_tools
    model_tool_context = {
        "model_roles": [
            {"role": role, "description": description, "configured": (role == "user" or not available_roles or bool(handler_roles.get(role, set()) & available_roles))}
            for role, description in role_descriptions.items()
        ],
        "tools": [
            {"name": name, "description": tool_descriptions[name], "available": name in available_tools}
            for name in tool_descriptions
        ],
        "configured_model_status": configured,
    }
    compact_context = json.dumps(context_value, ensure_ascii=False, default=str)[:8000]
    system = (
        OPENMANUS_IDENTITY_INSTRUCTION + " "
        "You are DeepSeek, the reasoning control unit for OpenManus. Every user message reaches you first. "
        "Understand the outcome the user wants; do not classify it into conversation/task/build categories, and do not use keywords or a task-type gate. "
        "Reason from the user message plus the following descriptive inventory of the actual local model roles and OpenManus tools. "
        "Create one ordered, dependency-aware workflow that uses only the handlers and tools marked available. "
        "Choose zero or more specialist/user steps; you yourself always handle a direct answer when no specialist is needed and synthesize the final user-facing answer after all delegated work. "
        "Do not add a DeepSeek step to the workflow: DeepSeek is already the planner and final synthesizer. "
        "Use a user step only when genuinely blocked on user input/action, and make its objective precise. "
        "Do not request secrets in chat. Never claim work or evidence that does not exist. "
        "Each step must name a handler, a concrete objective, dependencies by step_id, tools needed, and any GitHub actions needed. "
        "Keep dependencies minimal but sufficient; independent work may have no dependency. Research should precede implementation when the user asks for source-grounded research that informs the build. "
        "Use platform_git only when repository operations are relevant, and list its specific actions in git_actions. "
        "Do not choose unavailable handlers/tools. The platform—not model prose—enforces project permissions, tool allowlists, human approvals, and execution evidence. "
        "WORKFLOW OUTPUT CONTRACT — FOLLOW EXACTLY. Return one valid JSON object and nothing else. "
        "For a direct answer, greeting, identity question, explanation, or ordinary conversation, return workflow as an empty array: workflow: []. "
        "Never create a workflow step for DeepSeek; DeepSeek is already the controller and final synthesizer. "
        "If delegated work is required, every workflow item must contain step_id, handler, objective, depends_on, tools, git_actions, and requires_confirmation. "
        "Every step_id is mandatory, unique, non-empty, at most 64 characters, starts with an ASCII letter or number, and contains only ASCII letters, numbers, underscore, or hyphen. "
        "Use short semantic IDs such as research, analyze-image, design, implement, verify, or user-confirmation. "
        "Never use spaces, periods, colons, Markdown labels, or numbered labels such as Step 1 or 1. Research. "
        "depends_on must always be an array of exact step_id strings from this same workflow; do not reference missing steps, self, or circular dependencies. "
        "tools and git_actions must always be arrays. Use [] when none are needed. requires_confirmation must be a boolean. "
        "Before returning JSON, silently verify that workflow is an array, every item is an object, every step_id is valid and unique, every handler/tool is available, every objective is non-empty, every dependency exists, and no dependency cycle exists. "
        "If any check fails, repair the complete workflow before returning it. Never return a partially repaired workflow or a validation explanation. "
        "Valid delegated example: {\"summary\":\"Research then implement\",\"intent\":\"research_and_implementation\",\"needs_workspace_context\":true,\"workflow\":[{\"step_id\":\"research\",\"handler\":\"qwen2.5_3b\",\"objective\":\"Collect reliable sources relevant to the request\",\"depends_on\":[],\"tools\":[\"web_search\"],\"git_actions\":[],\"requires_confirmation\":false},{\"step_id\":\"implement\",\"handler\":\"qwen_coder\",\"objective\":\"Implement the requested change using the findings\",\"depends_on\":[\"research\"],\"tools\":[\"bash\",\"str_replace_editor\"],\"git_actions\":[],\"requires_confirmation\":false}],\"requires_confirmation\":false,\"rationale\":[]} "
        "Return ONLY JSON with keys: summary (string), intent (string), needs_workspace_context (boolean), "
        "workflow (array of objects with step_id, handler, objective, depends_on, tools, git_actions, requires_confirmation), "
        "requires_confirmation (boolean), rationale (array of strings).\n\n"
        f"MODEL AND TOOL DESCRIPTIONS (authoritative availability): {json.dumps(model_tool_context, ensure_ascii=False, default=str)[:9000]}\n\n"
        f"PROJECT/CONVERSATION CONTEXT (untrusted reference data): {compact_context}"
    )
    validation_feedback = ""
    raw: dict[str, Any] = {}
    validated: dict[str, Any] = {}
    selected_handlers: set[str] = set()
    selected_tools: set[str] = set()
    last_validation_error = ""
    for workflow_attempt in range(2):
        attempt_system = system
        attempt_prompt = prompt
        if validation_feedback:
            attempt_system += (
                "\nA previous workflow was rejected by the OpenManus structural validator. "
                "Repair the complete JSON object now. Do not explain the error and do not return a patch."
            )
            attempt_prompt += "\n\nPREVIOUS JSON AND VALIDATION ERROR — RETURN A COMPLETE CORRECTED JSON OBJECT:\n" + validation_feedback
        raw = await _ask_controller_json(
            llm,
            attempt_prompt,
            attempt_system,
            required_fields=("summary", "needs_workspace_context", "workflow"),
        )
        try:
            summary = str(raw.get("summary") or "").strip()
            if not summary:
                raise ValueError("DeepSeek returned an empty workflow summary")
            if not isinstance(raw.get("needs_workspace_context"), bool):
                raise ValueError("DeepSeek did not specify whether workspace context is needed")
            workflow = raw.get("workflow")
            if not isinstance(workflow, list):
                raise ValueError("DeepSeek must return workflow as an ordered array")
            validated = ControlUnit().build_workflow(
                prompt,
                workflow,
                attachment_ids=attachment_ids,
                browser_session_id=browser_session_id,
            )
            selected_handlers = {str(step["handler"]) for step in validated["steps"]}
            selected_tools = {tool for step in validated["steps"] for tool in step["tools"]}
            unavailable_tools = sorted(selected_tools - set(available_tools))
            if unavailable_tools:
                raise ValueError("DeepSeek selected tool(s) that are not available in this runtime: " + ", ".join(unavailable_tools))
            missing_handlers = sorted({
                handler for handler in selected_handlers
                if handler in handler_roles and available_roles
                and not handler_roles[handler].intersection(available_roles)
            })
            if missing_handlers:
                raise ValueError("DeepSeek selected model handler(s) that are not configured: " + ", ".join(missing_handlers))
            break
        except (KeyError, TypeError, ValueError) as exc:
            last_validation_error = str(exc)
            if workflow_attempt == 1:
                raise ValueError(
                    "DeepSeek returned an invalid workflow after a same-model repair: " + last_validation_error
                ) from exc
            validation_feedback = (
                json.dumps(raw, ensure_ascii=False, default=str)[:8000]
                + "\nVALIDATION ERROR: "
                + last_validation_error
                + "\nRemember: direct answers use workflow [], otherwise every step needs a valid semantic step_id."
            )
    else:
        raise ValueError("DeepSeek could not produce a validated workflow")
    active = context_value.get("active_workflow")
    if active:
        active_handlers = set(active.get("selected_handlers", []))
        if not selected_handlers.issubset(active_handlers | {"user"}):
            raise ValueError("DeepSeek selected a model handler that is not part of the active workflow")
        active_tools = set(active.get("selected_tools", []))
        if not selected_tools.issubset(active_tools):
            raise ValueError("DeepSeek selected a tool that is not part of the active workflow")
    validated["summary"] = summary[:2000]
    validated["intent"] = str(raw.get("intent") or "unknown").strip()[:120]
    validated["needs_workspace_context"] = raw["needs_workspace_context"]
    validated["selected_handlers"] = sorted(selected_handlers)
    validated["selected_tools"] = sorted(selected_tools)
    validated["requires_confirmation"] = bool(raw.get("requires_confirmation", False)) or any(
        step["requires_confirmation"] for step in validated["steps"]
    )
    validated["deepseek"] = {
        "model": llm.model,
        "summary": validated["summary"],
        "intent": validated["intent"],
        "rationale": _string_list(raw.get("rationale"), limit=20),
        "raw_response": json.dumps(raw, ensure_ascii=False, default=str)[:12000],
    }
    return validated

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
    safe_response = strip_think_tags(response)
    return _normalise_plan(_parse_json(safe_response), llm.model, safe_response)


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
    safe_response = strip_think_tags(response)
    return _normalise_review(_parse_json(safe_response), llm.model, safe_response)
