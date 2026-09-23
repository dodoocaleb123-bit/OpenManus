# Deploying the OpenManus Platform

The platform is a single container: FastAPI API + web UI + the Manus agent
runtime, with SQLite and per-project git checkouts on a persistent disk.
Everything below is driven by two files:

| File | Purpose |
| --- | --- |
| [`render.yaml`](render.yaml) | Render Blueprint: service, plan, region, disk, environment variables |
| [`runtime/Dockerfile`](runtime/Dockerfile) | Production image (Python 3.12, headless Chromium, lean dependency set) |

## Deploy to Render

### Prerequisites

- A [Render](https://render.com) account **with a payment method** — the
  persistent disk (and therefore durable projects/task history) requires a paid
  compute plan. The default `0.5c-512mb` plan is about $7/month plus ~$1.25/month
  for the 5 GB disk.
- This repository pushed to GitHub and your GitHub account connected to Render.
- An LLM API key (Anthropic, OpenAI, Azure OpenAI, Bedrock, or any
  OpenAI-compatible endpoint).
- Optional: a GitHub personal access token with the `repo` scope, for repository
  discovery, cloning, and pushing from the platform.

### Steps

1. In the Render Dashboard choose **New → Blueprint** and select this repository
   and the branch you want to deploy. Render detects `render.yaml` automatically.
2. Fill in the values Render prompts for:

   | Variable | What to enter |
   | --- | --- |
   | `LLM_MODEL` | Model name, e.g. `claude-sonnet-4-5` or `gpt-4.1` |
   | `LLM_BASE_URL` | Provider endpoint — see the table below |
   | `LLM_API_KEY` | Your provider API key |
   | `GITHUB_TOKEN` | PAT with `repo` scope (may be left empty to skip GitHub features) |

3. Click **Apply**. The first build takes roughly 5–10 minutes (it downloads
   Chromium); later builds reuse the cached dependency layer.
4. Open `https://<service-name>.onrender.com`. The browser asks for a username
   and password: user `admin`, password from **your service → Environment →
   `PLATFORM_PASSWORD`** (Render generated it for you).
5. Verify: `https://<service-name>.onrender.com/api/health` returns
   `{"status":"ok","service":"openmanus-platform"}` — this endpoint is the only
   one that does not require credentials, because Render's health checker polls it.

Every push to the deployed branch triggers a new deploy (`autoDeployTrigger: commit`).

### LLM provider settings

| Provider | `LLM_BASE_URL` | Extra variables |
| --- | --- | --- |
| Anthropic | `https://api.anthropic.com/v1/` | — |
| OpenAI | `https://api.openai.com/v1` | — |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | `LLM_API_TYPE=azure`, `LLM_API_VERSION=2024-08-01-preview` |
| Amazon Bedrock | `bedrock-runtime.<region>.amazonaws.com` | `LLM_API_TYPE=aws` plus AWS credentials as env vars; `LLM_API_KEY` can be any placeholder |
| OpenAI-compatible (PPIO, Jiekou.AI, ...) | provider's OpenAI-compatible URL | see `runtime/config/config.example-model-*.toml` |

Optional tuning: `LLM_MAX_TOKENS` (default 8192), `LLM_TEMPERATURE` (default 0.0),
`LLM_MAX_INPUT_TOKENS`, and `LLM_VISION_MODEL` (+ `LLM_VISION_BASE_URL` /
`LLM_VISION_API_KEY`) to add an `[llm.vision]` section.

Need more than `[llm]` (proxy, MCP servers, sandbox settings)? Add a Render
**Secret File** named `config.toml` with a full configuration based on
`runtime/config/config.example.toml`. It is mounted at `/etc/secrets/config.toml`
and takes precedence over the `LLM_*` variables.

### How configuration reaches the app

OpenManus reads `runtime/config/config.toml`, which is git-ignored and must never
be committed or baked into an image. At container start,
[`runtime/deploy/entrypoint.py`](runtime/deploy/entrypoint.py) writes that file
from the environment (or copies the secret file) and then starts the server. To
change models or keys, edit the environment variables in Render and redeploy;
the file is regenerated on every start.

### Sizing

Measured on this build: the server with the agent and all tools loaded idles at
about 120 MB RSS. A headless Chromium session adds roughly 150–300 MB.

- `0.5c-512mb` (default): fine for the API, agent tasks, and light use of the
  shared browser.
- `1c-2g`: choose this if you run browser sessions regularly, see
  "Out of memory" restarts in the Render logs, or run test suites of larger
  projects inside the coding loop.

Change `plan:` in `render.yaml` (or in the dashboard) — no other changes needed.

### Persistence

The disk is mounted at `/app/OpenManus/workspace`, which holds `platform.db`
(projects, tasks, events, checkpoints) and `projects/<name>/` git checkouts.
Render snapshots the disk daily. Services with disks run a single instance and
have a brief restart on each deploy; in-flight tasks are re-queued automatically
by the platform's restart recovery.

Active browser sessions are process-bound and are not restored after a restart
(existing platform limitation, see `runtime/README_PLATFORM.md`).

### Security notes

- **Keep `PLATFORM_PASSWORD` set.** The service URL is public. Without a password
  anyone can create projects, run agent tasks on your LLM budget, and use your
  `GITHUB_TOKEN`. Render terminates TLS, so credentials are never sent in clear
  text. Set `PLATFORM_USERNAME` to change the username (default `admin`).
- Agent tasks execute code (Python, shell, `pytest`) inside the service
  container. Use a dedicated GitHub token scoped to the repositories you intend
  to work on, and rotate it if the service is ever exposed without a password.
- Never paste keys into the repository, `render.yaml`, or chat — only into
  Render's Environment tab or a Secret File.

### Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Build fails around `playwright install --with-deps` | The image must stay on Debian 12 (`python:3.12-slim-bookworm`); Playwright 1.51 has no dependency list for Debian 13. |
| Log: `[entrypoint] LLM_API_KEY is set but LLM_MODEL ... is missing` | Set all three of `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`. |
| Log: `WARNING: no LLM configuration found` | The UI works but tasks will fail; add the `LLM_*` variables or a `config.toml` secret file. |
| Task fails immediately with an authentication error from the provider | Wrong `LLM_API_KEY`/`LLM_BASE_URL` pairing for the model. |
| `GITHUB_TOKEN is not configured` in the UI | Add the token in Environment (needs `repo` scope) and redeploy. |
| `git push` from a project fails | Token lacks write access to that repository. |
| Service restarts with OOM | Move to `1c-2g`. |
| Projects disappear after a deploy | The disk is missing or mounted somewhere other than `/app/OpenManus/workspace`. |

## Running the same image anywhere else

The image is not Render-specific. On any Docker host:

```bash
cd runtime
docker build -t openmanus-platform .
docker run -d --name openmanus -p 8000:8000 \
  -v openmanus-data:/app/OpenManus/workspace \
  -e LLM_MODEL=claude-sonnet-4-5 \
  -e LLM_BASE_URL=https://api.anthropic.com/v1/ \
  -e LLM_API_KEY=... \
  -e GITHUB_TOKEN=... \
  -e PLATFORM_PASSWORD=choose-a-strong-password \
  openmanus-platform
```

Put a TLS-terminating reverse proxy (Caddy, nginx, Traefik) in front of it
before exposing it to the internet; HTTP Basic auth is only safe over HTTPS.

## Local development (unchanged)

```bash
cd runtime
pip install -r requirements.txt   # or the lean requirements-platform.txt
playwright install chromium
cp config/config.example.toml config/config.toml   # add your key
python run_platform.py            # http://127.0.0.1:8000, no auth unless PLATFORM_PASSWORD is set
```
