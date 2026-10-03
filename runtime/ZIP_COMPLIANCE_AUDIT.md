# OpenManus DeepSeek workflow audit

**Scope:** Review the current registry-free implementation and its sandbox checks. This document describes source behavior; it does not claim that the user's Windows/Ollama deployment has been tested end to end.

## User-message path

1. The composer sends the user's original non-empty message and explicitly selected current-message attachment IDs to `POST /api/projects/{project_id}/chat`. Authentication, project access, request bounds, attachment ownership, concurrency, idempotency, and protected-action checks remain platform responsibilities.
2. For a new request, `runtime/app/api/routes.py` passes the original text directly to `make_authoritative_plan` in `runtime/app/platform/reasoning.py`. DeepSeek receives descriptive local model-role and runtime-available tool information. There is no greeting/math/build classifier, `requires_task` prefilter, conversation/task mode, numbered task catalog, or `respond`/`execute` route field.
3. DeepSeek authors zero or more descriptive workflow steps: handler, objective, dependencies, selected tools, optional Git actions, and confirmation requirement. An empty workflow means DeepSeek answers directly; a mixed workflow may include research, vision, design, coding, and a user action. The same DeepSeek model synthesizes the final user-facing answer after delegated work.
4. The platform validates workflow shape, dependencies, real handler/tool availability, project authorization, protected-action approvals, and completion evidence after DeepSeek chooses the workflow. Those execution/security checks do not decide what kind of request the user made.
5. Chat, scheduled automation, and active-task follow-ups all use the same planner. A follow-up may continue only handlers/tools already in the active task; it is still sent to DeepSeek, but the platform prevents an in-progress task from acquiring a newly expanded execution scope.

## Removed and retained surfaces

- Removed: `runtime/app/platform/capability_registry.py`, its 400-entry `capabilities.json`, ID-based workflow creation/validation, and `/api/capabilities/registry`.
- Retained: human-readable model role profiles and their status/health UI/API, the declarative plugin-manifest system, ordinary platform authorization and policy checks, and typed handoff/evidence structures. These are not consulted as a numbered task-selection catalog.
- The default execution path remains local Ollama. DeepSeek is mandatory for planning; the platform fails closed rather than substituting a keyword classifier or another model.

## Verification boundaries

The pytest suite and static checks can verify planner inputs/outputs, workflow validation, ordering, tool scoping, events, and mocked specialist handoffs. They cannot prove the configured Ollama tags exist, the target laptop can keep the models resident within its memory budget, browser/search connectivity works, or that a real multi-step task succeeds on that machine. Do not describe the change as laptop-verified until those checks have run on the user's Windows/Ollama deployment.

## Sandbox verification — 2026-10-03

- `pytest -q --ignore=tests/sandbox`: **195 passed**. The Docker-dependent sandbox tests were excluded; earlier runs confirmed this execution environment does not provide the Docker daemon those tests require.
- Python `compileall`, `node --check` on all 3 inline JavaScript blocks, and `git diff --check`: passed.
- No real Ollama/DeepSeek calls or Windows laptop deployment were performed. Model quality, memory pressure, and actual local inference latency remain to be verified on the target machine.
