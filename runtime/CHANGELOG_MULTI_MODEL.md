# Multi-model capability update

## Comparison pass / 2026-10-01

- Added explicit input/output schemas and supporting-handler metadata to all 400 capability records.
- Added typed five-role model profiles and corrected role-aware health reporting.
- Enforced bounded specialist inference concurrency and model request timeouts.
- Added a DeepSeek control-unit preflight to direct conversational routing before final model selection.
- Added the repeatable `scripts/integration_multimodel.py` Docker/Ollama contract check.

## Unreleased / 2026-10-01

Implemented the roadmap in `OpenManus Multi-Model Capability Update.docx`:

- Added five explicit Ollama role profiles with CPU-first sequential execution defaults.
- Added safe model-role status and non-generative Ollama health checks.
- Added `capabilities.json`, a versioned registry containing IDs 1–400 exactly once.
- Assigned Qwen Coder, Gemma 3, DeepSeek, Qwen2.5 3B, Llama 3.2 3B, platform services, and the user according to the roadmap.
- Added typed handoff requests/results, dependency ordering, evidence requirements, and bounded failure behavior.
- Added the DeepSeek control-unit plan to chat/task metadata and execution context.
- Bound plans to current-message attachment IDs.
- Added normalized design skill resources for the creativity specialist.
- Added Ollama Docker Desktop deployment instructions and a secret-free `.env.example`.
- Added regression tests for registry completeness, routing, handoffs, evidence, API surfaces, and control-unit planning.

The supplied capability document contains one extra conversational greeting after its claimed 400 capabilities. That sentence is not included as a capability; the registry preserves the intended IDs 1–400 exactly.

## Verification

- Non-Docker platform suite: 176 passed in the latest comparison pass.
- Focused multi-model/routing suite: 39 passed in the latest comparison pass.
- Python compilation: passed.
- Registry completeness: passed.
- Route smoke checks: passed.
- Docker sandbox tests remain environment-dependent because this execution environment has no Docker daemon; run them on the laptop during deployment validation.
