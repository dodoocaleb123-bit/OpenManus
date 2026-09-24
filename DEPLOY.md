# Deploying the OpenManus Platform

This guide puts **your** OpenManus Platform online at a real web address, behind a
username and password, on [Railway](https://railway.com). It is deliberately a
**single-user** deployment: everything on it runs as you, with your model key and
your GitHub token.

> **Not included on purpose:** public signups. Opening the platform to strangers
> requires per-user accounts, per-user GitHub connections and isolated agent
> sandboxes (see `runtime/BUILD_PLAN.md`). Until then, anyone with access to the
> deployed site acts as you.

## What you need

1. **A Railway account** — https://railway.com (sign in with GitHub).
2. **A model API key** — whatever works in your local `runtime/config/config.toml`
   today (Anthropic, OpenAI, Azure, Ollama, ...). This is billed separately by
   that provider.
3. **(Optional) a GitHub token** — only if you want the platform to list/clone/push
   repositories. Use a **fine-grained personal access token** limited to the
   repositories you need. Never a classic token with access to all your repos.

You do these steps yourself in the Railway dashboard; nobody else needs your keys.

## Steps

### 1. Create the service

1. Railway → **New Project** → **Deploy from GitHub repo** → pick `OpenManus`.
2. Open the service → **Settings → Root Directory** → set it to `runtime`.
   This makes Railway build `runtime/Dockerfile` (and ignore `training/`).

The first build downloads Chromium for the shared browser and takes several
minutes.

### 2. Set the variables

In the service's **Variables** tab (Raw Editor is quickest):

| Variable | Required | Example | Purpose |
| --- | --- | --- | --- |
| `PLATFORM_DATA_DIR` | yes | `/data` | Where the SQLite database and project workspaces live (= the volume mount path) |
| `PLATFORM_ADMIN_USER` | no | `admin` | Login name (default `admin`) |
| `PLATFORM_ADMIN_PASSWORD` | yes | a long random string | **Login password. If unset, the site runs without any login.** |
| `OPENMANUS_LLM_API_KEY` | yes | | Your model provider key |
| `OPENMANUS_LLM_MODEL` | yes | e.g. `claude-3-7-sonnet-20250219` | Same value as `model` in your working `config.toml` |
| `OPENMANUS_LLM_BASE_URL` | yes | e.g. `https://api.anthropic.com/v1/` | Same value as `base_url` in your `config.toml` |
| `OPENMANUS_LLM_API_TYPE` | if your `config.toml` sets it | e.g. `azure` | Provider type |
| `OPENMANUS_LLM_API_VERSION` | if your `config.toml` sets it | e.g. `2024-08-01-preview` | Azure API version |
| `OPENMANUS_LLM_MAX_TOKENS` | no | `8192` | Max tokens per request |
| `OPENMANUS_LLM_TEMPERATURE` | no | `0.0` | Sampling temperature |
| `OPENMANUS_VISION_MODEL` / `_BASE_URL` / `_API_KEY` | no | | Vision model settings for screenshots (defaults to the main model settings) |
| `GITHUB_TOKEN` | no | fine-grained PAT | Repository discovery, clone, push |
| `OPENMANUS_DISABLE_BROWSER_USE` | no | `1` | Recommended in containers: skips the optional browser-use MCP helper |
| `PORT` | automatic | Railway sets it | The server binds `0.0.0.0:$PORT` |

Environment variables always win over `config.toml`, so you can keep secrets off
disk entirely. The values are exactly the fields of your `[llm]` section.

### 3. Add the persistent volume

**Settings → Volumes → Add Volume**, mount path `/data` (5 GB on the Hobby plan,
resizable later). This must match `PLATFORM_DATA_DIR`. Without it, the database
and every project workspace are wiped on each deploy.

Note: redeploying a service with a volume causes a few seconds of downtime.

### 4. Health check and domain

1. **Settings → Deploy → Health Check Path**: `/api/health`.
2. After the first successful deploy: **Settings → Networking → Generate Domain**.
3. Open the domain — your browser asks for the username/password from step 2.

### 5. Cap your spend

Railway bills for actual usage and does not cap itself. Under **Billing**, set a
**usage cap** so a runaway agent task cannot produce a surprise bill.

## What it costs (September 2026)

Railway is usage-based: the Hobby plan is $5/month including $5 of usage, then
$0.000463 per vCPU-minute ($20/vCPU-month) and $0.000231 per GB-minute
($10/GB-month) [[1]](https://www.budgetforge.dev/tools/railway-pricing-2026),
plus $0.15/GB-month for the volume and $0.05/GB egress
[[2]](https://www.srvrlss.io/provider/railway/). A lightly used platform (agent
runs a few tasks a day) typically lands around **$10–20/month total**, dominated
by the always-on memory. Your model API usage is billed by your model provider.

For comparison: a 2 GB Render instance is $25/month
[[3]](https://comparedge.com/tools/render/performance); Fly.io prices are
similar to Railway [[4]](https://fly.io/pricing/).

## Security notes

- **Login**: everything except `/api/health` is behind HTTP Basic auth with
  `PLATFORM_ADMIN_USER` / `PLATFORM_ADMIN_PASSWORD`. Use a long password and
  HTTPS (Railway's generated domain provides TLS).
- **Your GitHub token is never visible to the agent.** It is scrubbed from the
  environment of code the agent runs (python/bash tools and workspace test
  commands). Pushing is done through the platform's Git controls, which keep the
  token server-side. Trade-off: `git push` from inside the agent's shell will not
  have credentials.
- **Model keys** live only in Railway variables. Keep the GitHub repo public-safe:
  never commit real keys to `config.toml` (it is gitignored — keep it that way).
- The agent executes code **inside the platform container** (no sandbox yet). You
  are trusting the tasks you give it, exactly as when running locally.

## Data and backups

Everything durable is under `/data`: `platform.db` (projects, tasks, events) and
`projects/` (each project's workspace). Railway volumes support manual and
automated backups — set a backup schedule after your first real project.

## Local production-style run

```bash
cd runtime
docker build -t openmanus-platform .
docker run --rm -p 8000:8000 \
  -e PLATFORM_ADMIN_USER=admin \
  -e PLATFORM_ADMIN_PASSWORD='pick-something-long' \
  -e PLATFORM_DATA_DIR=/data \
  -e OPENMANUS_LLM_API_KEY=... \
  -e OPENMANUS_LLM_MODEL=... \
  -e OPENMANUS_LLM_BASE_URL=... \
  -v openmanus-data:/data \
  openmanus-platform
```

Without Docker: `pip install -r requirements.txt && playwright install chromium`,
then `python run_platform.py` (honours `HOST`, `PORT`, `PLATFORM_DATA_DIR` and all
variables above).

## Known limits

- Single user, single node. No signups, teams or per-user GitHub accounts.
- Live task streams auto-reconnect and heart-beat (Railway closes idle streams
  after 5 minutes and any stream after 15 minutes
  [[5]](https://docs.railway.com/networking/public-networking/specs-and-limits));
  a task that runs longer is still fine — reconnect resumes the feed.
- Tasks interrupted by a redeploy are re-queued at startup (in the background).
- `ask_human` tool output goes to the server logs, not the web UI.
- Browser sessions are not restored after a server restart (project metadata is).
- Node validation assumes `npm test` works in the project workspace.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Browser asks for password repeatedly | Check `PLATFORM_ADMIN_PASSWORD` is set and the username matches `PLATFORM_ADMIN_USER` |
| UI shows "GitHub: GITHUB_TOKEN not connected" | Set `GITHUB_TOKEN` (fine-grained PAT) in Variables |
| Task fails immediately with a model error | Compare `OPENMANUS_LLM_*` with the `[llm]` values of your working local `config.toml` |
| Build fails at `playwright install` | Retry the deploy (CDN hiccup); if it persists, remove that line from `Dockerfile` (disables the shared browser) |
| Changes lost after deploy | The volume is missing or `PLATFORM_DATA_DIR` ≠ volume mount path |
| Node projects always fail validation | `npm test` must work inside the workspace (deps installed by the agent) |
