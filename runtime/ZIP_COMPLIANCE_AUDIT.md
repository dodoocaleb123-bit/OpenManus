# OpenManus capability update — implementation audit

**Scope:** review branch `review/zip-multimodel-audit-20261001`. This file records the current code contract and known limits; it is not a claim that the user's local Ollama deployment has passed end-to-end acceptance.

The code review covered the user-supplied ZIP contents previously inventoried in this file: `New update.docx`, `OpenManus Multi-Model Capability Update.docx`, `OpenManus capabilities categorized into eight categories.docx`, and the included design skill files. The numbered capability inventory remains the source of truth where illustrative examples use different IDs. External Taste, Impeccable, UI/UX Pro Max resources, and Sleek's optional Pro API are not bundled and are not part of the zero-cost local default.

## User-message path

1. The project composer sends the original message and explicit current-message attachment IDs to `POST /api/projects/{project_id}/chat`. Authentication, request bounds, project ownership, attachment validation, and idempotent replay remain platform-level checks; replaying the same idempotency key returns the saved task rather than repeating model work.
2. For every newly accepted message, `runtime/app/api/routes.py` calls `make_authoritative_plan` in `runtime/app/platform/reasoning.py`. The unchanged user text goes to the configured local DeepSeek model. There is no greeting/math/build keyword classifier, `requires_task` prefilter, user-selected answer/build mode, or `respond`/`execute` route field.
3. DeepSeek first selects the model handler roles needed for the complete outcome, then selects exact ordered capability IDs from the executable directory for those roles. The platform validates schema, IDs, selected handlers, configured local model roles, dependencies, permissions, and required evidence **after DeepSeek's decision**. Each accepted composer message becomes a persisted task, including ordinary questions and greetings; user-facing answering is a selected DeepSeek capability in that task.
4. The orchestrator runs selected specialist steps and passes their structured results and available evidence to the selected DeepSeek output capability. Dependencies ensure that final output waits for prior research, vision, design, coding, or user-action steps. Ollama JSON-mode compatibility has a bounded same-DeepSeek retry for malformed/think-only control output; no alternate model is substituted.
5. Scheduled prompts use the same controller. Active-task follow-ups also go through DeepSeek, but the current endpoint only continues capability steps already selected for that running task; it rejects a newly selected capability workflow until the task ends. Explicit Stop/confirmation UI controls remain platform actions.

## Capability and model coverage

| Area | Current implementation | Limitation / verification status |
| --- | --- | --- |
| DeepSeek-first planning | Handler selection followed by capability-ID selection for every accepted new request; unified task response contract | Actual DeepSeek-R1 7B quality and latency on the user's laptop are not verified in this sandbox. |
| Multi-model workflow | Qwen Coder, Gemma 3, Qwen2.5 3B, Llama 3.2 3B, DeepSeek, and supported user-action steps have local roles; final DeepSeek synthesis is in the same workflow | The registry contains 400 catalog entries, but not 400 distinct executable adapters. Unsupported UI/platform inventory entries are not executable capabilities. |
| Research / build / answer | Research evidence, vision observations, design brief, and coding results can be ordered before final DeepSeek synthesis | Web search depends on network and provider availability; factual completeness is not guaranteed. Test fixtures use deterministic local fakes. |
| Images | Only explicitly attached current-message image IDs are sent to Gemma | Unsupported or missing image inputs fail visibly; prior project images are not implicitly reattached. |
| User actions | A DeepSeek-selected user step pauses, records the reply, and resumes; reply acknowledgement is not external proof | Login, CAPTCHA, payment, and other external actions still need platform/browser verification where applicable. |
| GitHub | Tool exposure is capability-scoped; protected mutations require authorization/confirmation and matching action evidence | Live GitHub execution was not performed in the sandbox. |
| Local operation | Default path is local Ollama and zero-cost model execution, without requiring a hosted LLM API | Five model tags, Docker memory use, model health, UI responsiveness, and laptop end-to-end timing require validation on the target Windows/Ollama machine. |

## Acceptance caveats

A passing pytest suite can verify the code contract, event ordering, capability dependencies, and mocked specialist handoffs. It cannot prove the configured Ollama tags exist, the laptop can keep the models resident within 16 GB, browser/search access works, or the full screenshot → research → design → build → preview/review experience meets the ZIP's acceptance criteria. Do not describe the update as laptop-verified until those checks have actually run on the user's machine.


## Sandbox verification — 2026-10-02

- `pytest -q --ignore=tests/sandbox`: **199 passed**. The output-capability tests use local fakes; this does not test real Ollama.
- Full `pytest -q` was also attempted. The 18 sandbox-test errors and 8 sandbox-test failures stem from the unavailable Docker daemon in this execution environment; the initial run also exposed three v07 tests calling a real configured output model. Those tests now use the offline DeepSeek fake, and their targeted rerun passed (**3 passed**).
- `python3 -m compileall -q app`, `node --check` for all three inline JavaScript blocks, and `git diff --check`: passed.
- Browser smoke test: the task-only composer loaded, a disposable project was selected, a mocked `/chat` response created a task view, and a simulated `assistant.delta` token rendered incrementally. No live DeepSeek/Ollama response was attempted because no local model is configured in this sandbox.
