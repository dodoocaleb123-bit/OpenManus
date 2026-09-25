# OpenManus Platform — production image.
#
# One container = the platform (FastAPI API + web UI) + the agent's "computer":
# Python 3.11, Node.js 22 + npm, git, a C/C++ toolchain and headless Chromium.
# The agent builds, runs and tests projects inside this container.
#
# Build context is this directory (runtime/). Referenced by ../render.yaml.
#
#   docker build -t openmanus-platform .
#   docker run -p 8000:8000 -v openmanus-data:/app/OpenManus/workspace \
#       -e LLM_MODEL=... -e LLM_BASE_URL=... -e LLM_API_KEY=... openmanus-platform

# Node.js comes from the official image (same Debian release, so the binary's
# glibc/libstdc++ match); copying it avoids a third-party apt repository.
FROM node:22-bookworm-slim AS node

# Pinned to bookworm: Playwright 1.51 targets Debian 12. Python 3.11 is used
# because it has compatible greenlet wheels across Docker Desktop architectures.
FROM python:3.11-slim-bookworm

# PLAYWRIGHT_BROWSERS_PATH  Chromium lives at a fixed, user-independent path.
# TIKTOKEN_CACHE_DIR        Encodings are pre-baked below so the first task needs no download.
# HOST / PORT               Bind on all interfaces; hosting platforms such as Render override PORT.
# OPENMANUS_DISABLE_BROWSER_USE
#                           The Browser Use MCP server needs `uvx`, which is not in this image;
#                           the platform's own shared browser (platform_browser) replaces it.
# GIT_*                     Identity used when the platform commits inside project workspaces
#                           (git refuses to commit without one). Override per deployment.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    HOST=0.0.0.0 \
    PORT=8000 \
    OPENMANUS_DISABLE_BROWSER_USE=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    GIT_AUTHOR_NAME="OpenManus Platform" \
    GIT_AUTHOR_EMAIL="openmanus@users.noreply.github.com" \
    GIT_COMMITTER_NAME="OpenManus Platform" \
    GIT_COMMITTER_EMAIL="openmanus@users.noreply.github.com"

WORKDIR /app/OpenManus

# git: project checkouts. build-essential: native npm/pip packages (the agent's compiler).
# procps: `ps`/`kill` for the agent to manage the dev servers it starts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates build-essential procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -sf ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && node --version && npm --version

# Dependencies first so this (slow) layer is cached across source-only changes.
# requirements-platform.txt is the lean runtime subset of requirements.txt; the
# full file drags in browsergym -> torch + CUDA (~4.6 GB) which the server never uses.
# `playwright install --with-deps` runs its own apt-get update for Chromium's libraries.
# app/llm.py uses the model's tiktoken encoding and falls back to cl100k_base.
COPY requirements-platform.txt ./
RUN pip install --prefer-binary --retries 10 --timeout 300 -r requirements-platform.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"

COPY . .

# SQLite database + per-project workspaces live here. Mount a volume or a
# persistent disk at this exact path to keep data across deploys/restarts.
RUN mkdir -p workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# The entrypoint writes config/config.toml from LLM_* env vars (or a mounted
# secret file) and then execs the command below.
ENTRYPOINT ["python", "deploy/entrypoint.py"]
CMD ["python", "run_platform.py"]
