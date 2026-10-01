# OpenManus multi-model Ollama deployment

OpenManus now uses a versioned registry of all 400 capabilities. DeepSeek is the control unit for planning and conversational routing; platform services remain authoritative for tools, persistence, evidence, and protected actions.

## Roles

| Role | Default Ollama model | Responsibility |
|---|---|---|
| Qwen Coder | `qwen2.5-coder:7b` | Software engineering, tests, previews, and repository work |
| Gemma 3 | `gemma3:4b` | Current-message image and screenshot analysis |
| DeepSeek | `deepseek-r1:7b` | Reasoning, planning, orchestration, and final synthesis |
| Qwen2.5 | `qwen2.5:3b` | Source-grounded web research |
| Llama 3.2 | `llama3.2:3b` | Creativity, content creation, and beautiful design briefs/reviews |

All five profiles use the same Ollama OpenAI-compatible endpoint by default. Copy `.env.example` to `.env` and change model tags to match `ollama list`.

The typed role definitions live in `app/platform/model_profiles.py`. Every registry record also carries input/output schemas and supporting-handler metadata. DeepSeek performs a control-unit preflight for direct conversation, while image replies remain on Gemma so image bytes reach a vision-capable model.

## CPU and memory policy

The default is one active model at a time (`PLATFORM_MAX_MODEL_CONCURRENCY=1`). Do not enable parallel specialist execution until the laptop has been tested for peak RAM, model unload time, and preview responsiveness. The platform records model role status at `/api/capabilities/models/status` and the complete capability map at `/api/capabilities/registry`.

## Docker Desktop on Windows

From PowerShell in the repository root:

```powershell
cd runtime
docker build -t openmanus-platform .
docker stop openmanus 2>$null
docker rm openmanus 2>$null
docker run -d --name openmanus --restart unless-stopped `
  --add-host=host.docker.internal:host-gateway `
  --env-file ..\.env `
  -p 127.0.0.1:8000:8000 `
  -v openmanus-data:/app/OpenManus/workspace `
  openmanus-platform
```

Before starting, run `ollama list` and confirm all configured tags exist. Keep `.env` outside Git tracking. Open `http://127.0.0.1:8000` and check `/api/health`, `/api/capabilities/models/status`, and `/api/capabilities/registry`.

After the container starts, run the redacted contract check from `runtime`:

```powershell
python scripts/integration_multimodel.py
```

For a complete laptop acceptance run, verify that the health endpoint reports `model_present: true` for all five roles, then exercise an image question, a source-grounded research request, a design-only request, a design-and-build request, and a protected GitHub operation. Record task IDs and evidence events; model output alone is not completion proof.

## Workflow guarantees

- Research-only requests cannot create project files unless the user asks for an artifact.
- Design-only requests do not invoke Qwen Coder.
- Image attachments are bound to the current message and routed to Gemma 3.
- Model output alone is never treated as proof; evidence must come from files, tests, browser state, source metadata, or platform responses.
- Passwords, MFA, CAPTCHA, payment, and private authentication remain user actions.
- GitHub tokens remain inside the protected integration.
- Failed steps are checkpointed and bounded retries/repairs are used.
- Specialist inference is guarded by `PLATFORM_MAX_MODEL_CONCURRENCY` and `PLATFORM_MODEL_TIMEOUT_SECONDS`; the default is one active model request.
