# Deploying the OpenManus Platform to Render

The platform is **one container**: the web UI + API, the agent, and the agent's
"computer" (Python 3.12, Node.js 22 + npm, git, a C/C++ toolchain and headless
Chromium). Projects, task history and checkouts live on a persistent disk.

| File | Purpose |
| --- | --- |
| [`render.yaml`](render.yaml) | Render Blueprint: service, plan, region, disk, environment variables |
| [`runtime/Dockerfile`](runtime/Dockerfile) | Production image |
| [`runtime/deploy/entrypoint.py`](runtime/deploy/entrypoint.py) | Turns `LLM_*` env vars into `config/config.toml` at start-up |

**Monthly cost:** Render `1c-2g` instance **$25** + 10 GB disk **$2.50** (Render's
free Hobby workspace plan is fine), **plus your LLM provider's usage**. A typical
build task makes roughly 10–60 model calls.

---

## Step-by-step

### 1. Put the code on the branch Render will deploy

Merge the deployment pull request into `main` on GitHub (or pick the PR branch in
step 5 if you want to try it before merging). Render deploys from GitHub, not from
your computer.

### 2. Get an LLM API key (vision-capable model)

The agent reads screenshots of the apps it builds, so use a model that accepts
images. Recommended: **Anthropic**.

1. Go to <https://console.anthropic.com> → sign up → **Billing** → add credit.
2. **API Keys** → **Create Key** → copy it (starts with `sk-ant-`). You only see it once.
3. Pick a model ID from Anthropic's model list, e.g. `claude-sonnet-5`
   (good balance of cost and coding ability) or `claude-opus-5` (strongest, pricier).

You will need three values:

| Variable | Anthropic | OpenAI |
| --- | --- | --- |
| `LLM_MODEL` | `claude-sonnet-5` | a current model ID from platform.openai.com, e.g. `gpt-5` |
| `LLM_BASE_URL` | `https://api.anthropic.com/v1/` | `https://api.openai.com/v1` |
| `LLM_API_KEY` | `sk-ant-...` | `sk-...` |

Other providers are listed [further down](#llm-provider-settings).

### 3. Create a GitHub token (optional but recommended)

This lets the platform list your repos, clone them, push branches, open pull
requests and create new repos. Pick **one** of the options below.

**Option A: classic token (simplest).**

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
   → **Generate new token (classic)**.
2. Note: `OpenManus platform`. Expiration: your choice (90 days is sensible).
3. Scope: tick **`repo`** (that's all).
4. **Generate token** → copy it (starts with `ghp_`). Use it as `GITHUB_TOKEN`.

**Option B: fine-grained token (least privilege).**

Grant **Contents: read/write**, **Pull requests: read/write**, **Metadata: read**,
and **Administration: read/write** only if you want the agent to create new
repositories (that also needs "All repositories"). Copy it (starts with
`github_pat_`) and use it as `GITHUB_TOKEN`.

**Option C: both tokens (fine-grained first, classic as fallback).**

Create the fine-grained token from option B **and** the classic token from option A.

- `GITHUB_TOKEN` = the fine-grained token (`github_pat_...`). It is always tried first.
- `GITHUB_CLASSIC_TOKEN` = the classic token (`ghp_...`). It is used **only** when
  GitHub refuses the first one: API answers 401/403/404, or git reports an
  authentication/permission error on clone or push. Everything else (e.g. a
  rejected non-fast-forward push, a validation error) is not retried.

This is handy when the fine-grained token doesn't cover a repository (an org that
blocks fine-grained tokens, a repo you forgot to select) without giving up least
privilege for everything else.

How the platform handles the tokens, whichever option you pick:

- Tokens never appear in command lines, remote URLs, `.git/config` or logs
  (output is redacted), and are hidden from the agent's shell and Python tools.
- Git runs in a credential-free environment (no system/global git config, no
  injected `GIT_CONFIG_*`, no inherited askpass/SSH helpers). Only clone and push
  receive a token, through a temporary askpass helper that answers **only** for
  `https://github.com`.
- Pushes run with credential helpers and hooks disabled (`--no-verify`,
  `core.hooksPath=/dev/null`), and are refused if the project's `.git/config`
  contains risky settings (credential helpers, `core.hooksPath`/`sshCommand`/
  `fsmonitor`, `url.*.insteadOf`, `remote.*.pushurl`, `include.path`, `http.*`,
  ...). The error names the offending keys so you can `git config --unset` them.

### 4. Prepare Render

1. Sign up at <https://render.com> (use **Sign up with GitHub**, which is the easiest).
2. **Account → Billing**: add a payment method. The persistent disk requires a
   paid instance type.
3. If you signed up with email: **Account Settings → Git providers → Connect
   GitHub**, and give Render access to this repository.

### 5. Create the service from the Blueprint

1. Render Dashboard → **New +** → **Blueprint**.
2. Select this repository. Branch: `main` (or the PR branch).
3. Render reads `render.yaml` and shows one web service, `openmanus-platform`, with a
   disk. It asks for the secret values:

   | Prompt | Enter |
   | --- | --- |
| `LOCAL_LLM_MODEL` | Ollama model, e.g. `qwen2.5-coder:7b` |
| `LOCAL_LLM_BASE_URL` | Container-reachable Ollama endpoint, e.g. `http://host.docker.internal:11434/v1` |
| `VISION_LLM_MODEL`, `VISION_LLM_BASE_URL`, `VISION_LLM_API_KEY_01` … `_10` | Cloud vision model, endpoint, and 10-key pool |
| `HEAVY_CODING_LLM_MODEL`, `HEAVY_CODING_LLM_BASE_URL`, `HEAVY_CODING_LLM_API_KEY_01` … `_10` | Cloud heavy-coding model, endpoint, and 10-key pool |
| `OLLAMA_FALLBACK_LLM_MODEL`, `OLLAMA_FALLBACK_LLM_BASE_URL`, `OLLAMA_FALLBACK_LLM_API_KEY_01` … `_10` | Cloud fallback model, endpoint, and 10-key pool |
| `LLM_API_KEY`, `LLM_API_KEYS` | Legacy single-provider compatibility only; do not use in the new three-pool setup |
| `GITHUB_TOKEN` | from step 3 (leave empty to skip GitHub features) |
   | `GITHUB_CLASSIC_TOKEN` | option C only: the classic fallback token (otherwise leave empty) |

4. Click **Apply** (or **Deploy Blueprint**).

### Dedicated model pools and routing

The new local-first configuration has no primary cloud API-key pool. Use Ollama as the default provider and keep three separate cloud pools:

```env
LOCAL_LLM_MODEL=qwen2.5-coder:7b
LOCAL_LLM_BASE_URL=http://host.docker.internal:11434/v1
LOCAL_LLM_API_KEY=ollama

VISION_LLM_MODEL=gemini-3-flash-preview
VISION_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
VISION_LLM_API_KEY_01=vision-key-01
VISION_LLM_API_KEY_02=vision-key-02
VISION_LLM_API_KEY_03=vision-key-03
VISION_LLM_API_KEY_04=vision-key-04
VISION_LLM_API_KEY_05=vision-key-05
VISION_LLM_API_KEY_06=vision-key-06
VISION_LLM_API_KEY_07=vision-key-07
VISION_LLM_API_KEY_08=vision-key-08
VISION_LLM_API_KEY_09=vision-key-09
VISION_LLM_API_KEY_10=vision-key-10

HEAVY_CODING_LLM_MODEL=gemini-3-flash-preview
HEAVY_CODING_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
HEAVY_CODING_LLM_API_KEY_01=coding-key-01
HEAVY_CODING_LLM_API_KEY_02=coding-key-02
HEAVY_CODING_LLM_API_KEY_03=coding-key-03
HEAVY_CODING_LLM_API_KEY_04=coding-key-04
HEAVY_CODING_LLM_API_KEY_05=coding-key-05
HEAVY_CODING_LLM_API_KEY_06=coding-key-06
HEAVY_CODING_LLM_API_KEY_07=coding-key-07
HEAVY_CODING_LLM_API_KEY_08=coding-key-08
HEAVY_CODING_LLM_API_KEY_09=coding-key-09
HEAVY_CODING_LLM_API_KEY_10=coding-key-10

OLLAMA_FALLBACK_LLM_MODEL=gemini-3-flash-preview
OLLAMA_FALLBACK_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OLLAMA_FALLBACK_LLM_API_KEY_01=fallback-key-01
OLLAMA_FALLBACK_LLM_API_KEY_02=fallback-key-02
OLLAMA_FALLBACK_LLM_API_KEY_03=fallback-key-03
OLLAMA_FALLBACK_LLM_API_KEY_04=fallback-key-04
OLLAMA_FALLBACK_LLM_API_KEY_05=fallback-key-05
OLLAMA_FALLBACK_LLM_API_KEY_06=fallback-key-06
OLLAMA_FALLBACK_LLM_API_KEY_07=fallback-key-07
OLLAMA_FALLBACK_LLM_API_KEY_08=fallback-key-08
OLLAMA_FALLBACK_LLM_API_KEY_09=fallback-key-09
OLLAMA_FALLBACK_LLM_API_KEY_10=fallback-key-10
```

Each key is a separate `.env` entry, so the file stays readable. The parser also
still accepts the older `*_API_KEYS=key1,key2` format. Routing is: deterministic normal code first; then Ollama for tasks it can handle;
vision tasks use the vision pool; tasks classified as beyond Ollama use the heavy-coding
pool; and Ollama failures/timeouts switch to the fallback pool. Each pool rotates its
own keys. Keys are read only from environment variables, never written to Git, and
never displayed in the UI. Ten keys are not a quota bypass: keys belonging to the same
Google Cloud project generally share project-level RPM/TPM/RPD limits, so separate
projects provide better isolation.

### 6. Watch the first build (about 8–15 minutes)

Open the service → **Logs** / **Events**. The slow parts are installing Chromium and
the build tools. It's done when you see:

```
[entrypoint] Generated config/config.toml from LLM_* environment variables (model=claude-sonnet-5, ...)
[entrypoint] Starting: python run_platform.py
INFO:     Uvicorn running on http://0.0.0.0:8000
```

and the service status turns **Live**. Later deploys are faster: dependency layers
are cached.

### 7. Log in

1. Service page → **Environment** → reveal **`PLATFORM_PASSWORD`** (Render generated it).
2. Open `https://openmanus-platform-XXXX.onrender.com` (the URL at the top of the
   service page). Username **`admin`**, password from above.
3. The header shows two status pills: **`model: claude-sonnet-5 · vision`** and your
   GitHub login. If the model isn't configured, a yellow banner explains what's
   missing; fix that variable (step 9).

Health check (no password needed): `https://<your-url>/api/health` →
`{"status":"ok",...}`.

### 8. Run a first task (smoke test)

1. Under **Projects**, type `hello-app` into *New project name* → **Create**.
2. In the task box type:
   *"Create a small Node.js Express app with a /health route and a Jest test for it.
   Run the tests and commit when they pass."*
3. Watch the activity feed: the agent plans, writes files, runs `npm install` and
   `npm test`, and commits. The platform then **re-validates the project
   independently** and sends the agent back to fix any failure (up to 3 repair
   cycles). The task only shows **succeeded** when validation passes. Click
   **Run agent** to start.
4. If the agent asks a question, a reply box appears in the feed. Answer it there.

Then try GitHub (right panel → **GitHub** tab): **Publish** creates a private repo
and pushes the project; *Connect an existing repository* → **Load my repositories**
→ **Clone** brings one of yours into the selected project (use an empty project). After a task, use **Push** and
**Open PR**. You can also just ask the agent: *"publish this to GitHub"* or *"open a
pull request"*.

### 9. Changing settings later

Service → **Environment** → edit → **Save, rebuild and deploy** (or **Save and
deploy**). The config file is regenerated on every start, so model or key changes
take effect after the restart. Pushing to the deployed branch also redeploys
automatically.

---

## Using the platform

- **Left:** projects and task history. **Middle:** live activity feed (plan, tool
  calls, command output, validation, questions) and the message box. **Right:** the
  shared browser, plus GitHub and Files tabs.
- **Shared browser:** the agent opens the app it is building (e.g.
  `http://localhost:3000`) and you see the same page. Click on the screenshot to click
  in the page; type into it using the keyboard buttons. Use it to log in somewhere
  for the agent, or to point at a bug.
- **Messages while running:** anything you send is delivered to the agent at its
  next step. **Stop agent** cancels; **Retry** (in Task history) re-runs a failed or
  cancelled task in the same workspace.
- One task runs per project at a time. Different projects can run in parallel,
  though a `1c-2g` instance is comfortable with one or two.

## Agent tuning (Environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_MAX_STEPS` | 40 | Upper bound for adaptive agent steps; simple requests use at most 6 and normal requests at most 20 |
| `AGENT_MAX_REPAIR_CYCLES` | 3 | Validation-driven repair passes after implementation |
| `AGENT_COMMAND_TIMEOUT` | 300 | Seconds before a single shell command is interrupted |
| `HUMAN_REPLY_TIMEOUT` | 900 | Seconds the agent waits for your answer before continuing on its own |
| `LLM_MAX_TOKENS` | 8192 | Max output tokens per model call |
| `LLM_REQUEST_TIMEOUT` | 180 | Seconds allowed for one local or cloud model request before failover/retry |
| `LLM_SUPPORTS_IMAGES` | auto | `true`/`false` if vision support is misdetected for your model |
| `LLM_REASONING_MODEL` | auto | `true`/`false` if your model rejects `temperature`/`max_tokens` |
| `PLATFORM_USERNAME` | admin | Login username |

### Local-first hybrid routing

When `LOCAL_LLM_MODEL` and `LOCAL_LLM_BASE_URL` are configured, OpenManus uses Ollama for normal coding and conversation. Simple create-and-verify file requests use a deterministic fast path and do not call an LLM. Vision tasks use `VISION_LLM_*`; heavy coding uses `HEAVY_CODING_LLM_*`; and Ollama failures use `OLLAMA_FALLBACK_LLM_*`.

Set these variables when local-first mode is enabled:

| Variable | Purpose |
| --- | --- |
| `LOCAL_LLM_MODEL` | Local Ollama/OpenAI-compatible model, for example `qwen2.5-coder:7b` |
| `LOCAL_LLM_BASE_URL` | Local endpoint reachable from the container, for example `http://host.docker.internal:11434/v1` |
| `VISION_LLM_*` | Dedicated cloud vision pool (`MODEL`, `BASE_URL`, `API_KEY_01` through `API_KEY_10`) |
| `HEAVY_CODING_LLM_*` | Dedicated cloud heavy-coding pool (`MODEL`, `BASE_URL`, `API_KEY_01` through `API_KEY_10`) |
| `OLLAMA_FALLBACK_LLM_*` | Dedicated cloud fallback pool (`MODEL`, `BASE_URL`, `API_KEY_01` through `API_KEY_10`) |

The old `CLOUD_LLM_*` variables remain supported as a migration fallback, but the new setup should use only the three dedicated pools above.

## LLM provider settings

| Provider | `LLM_BASE_URL` | Extra variables |
| --- | --- | --- |
| Anthropic | `https://api.anthropic.com/v1/` | none |
| OpenAI | `https://api.openai.com/v1` | none |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | `LLM_API_TYPE=azure`, `LLM_API_VERSION=2024-08-01-preview` |
| Amazon Bedrock | `bedrock-runtime.<region>.amazonaws.com` | `LLM_API_TYPE=aws` plus AWS credentials as env vars; `LLM_API_KEY` can be any placeholder |
| OpenAI-compatible (OpenRouter, PPIO, ...) | provider's OpenAI-compatible URL | see `runtime/config/config.example-model-*.toml` |

The model must support **tool/function calling**. Text-only models work for
coding, but the agent can't look at screenshots; for those, add a separate vision
model with `LLM_VISION_MODEL` (+ `LLM_VISION_BASE_URL` / `LLM_VISION_API_KEY`).

Need more than `[llm]` (proxy, MCP servers)? Add a Render **Secret File** named
`config.toml` based on `runtime/config/config.example.toml`. It is mounted at
`/etc/secrets/config.toml` and takes precedence over the `LLM_*` variables.

## Sizing

The server idles at about 120 MB of memory. Chromium adds 150–300 MB, and `npm
install`, builds and test runners can take several hundred MB more at their peak.

- `1c-2g` ($25, default): one or two concurrent tasks on typical web projects.
- `2c-4g` ($85): larger projects, heavy builds, several concurrent tasks.
- `0.5c-512mb` ($7): **not recommended**. The kernel kills the whole service
  (including the platform) when an install or browser session exceeds 512 MB.

Change `plan:` in `render.yaml` or in the dashboard. The disk (`sizeGB`) can be
grown later but never shrunk.

## Persistence

The disk at `/app/OpenManus/workspace` holds `platform.db` (projects, tasks,
events, checkpoints) and `projects/<name>/` workspaces. Render snapshots it daily.
Services with a disk run a single instance and have a short outage on each deploy.
Tasks that were running are re-queued automatically when the service comes back
(`task.recovered` in the feed) and continue in the same workspace. Browser sessions
are in-memory; the agent reopens the page when it needs it.

## Security notes

- **Keep `PLATFORM_PASSWORD` set.** The URL is public. Without a password anyone
  could run tasks on your LLM budget and use your GitHub token. Render terminates
  TLS, so the password is never sent in clear text.
- The agent runs commands inside the container. API keys are **removed from the
  environment of every command it runs**, and `GITHUB_TOKEN` is only used by the
  platform's own git/GitHub calls, never written into project remotes. The
  generated `config/config.toml` (containing the LLM key) is still readable from
  inside the container. Treat what you ask the agent to do accordingly, and use
  a provider key with a spending limit.
- Use a GitHub token limited to what you need, and rotate it if it ever leaks.
- Never put keys in the repository, `render.yaml` or chat. Put them only in
  Render's Environment tab or a Secret File.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Build fails around `playwright install --with-deps` | The image must stay on `python:3.12-slim-bookworm` (Debian 12). |
| Setup banner: model not configured / `WARNING: no LLM configuration found` in logs | Set all three of `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, then redeploy. |
| Task fails at once with "authentication" / 401 | Wrong key, or key and base URL from different providers. |
| Task fails with "model not found" / 404 | `LLM_MODEL` is misspelled or not available to your account. |
| "does not support images" error | The model has no vision; pick a vision model, or set `LLM_SUPPORTS_IMAGES=true` if it does. |
| Rate-limit / 429 errors | Provider limits; the platform retries transient errors automatically. Upgrade your provider tier for heavy use. |
| `GITHUB_TOKEN is not configured` | Add the token in Environment and redeploy. |
| Push / PR / Publish fails with 403 | The token lacks `repo` scope or write access to that repository. |
| Service restarts, "Out of memory" in Events | Use `2c-4g`, or run fewer tasks at once. |
| Projects disappear after a deploy | The disk is missing or not mounted at `/app/OpenManus/workspace`. |
| Browser panel stays blank | Ask the agent to start the app's dev server and open it, or enter a URL in the browser bar. |

## Running the same image anywhere else

```bash
cd runtime
docker build -t openmanus-platform .
docker run -d --name openmanus -p 8000:8000 \
  -v openmanus-data:/app/OpenManus/workspace \
  -e LLM_MODEL=claude-sonnet-5 \
  -e LLM_BASE_URL=https://api.anthropic.com/v1/ \
  -e LLM_API_KEY=... \
  -e GITHUB_TOKEN=... \
  -e PLATFORM_PASSWORD=choose-a-strong-password \
  openmanus-platform
```

Put a TLS-terminating reverse proxy (Caddy, nginx, Traefik) in front before exposing
it to the internet; HTTP Basic auth is only safe over HTTPS.

## Local development

```bash
cd runtime
pip install -r requirements-platform.txt
playwright install chromium
cp config/config.example.toml config/config.toml   # add your model + key
python run_platform.py            # http://127.0.0.1:8000, no auth unless PLATFORM_PASSWORD is set
```
