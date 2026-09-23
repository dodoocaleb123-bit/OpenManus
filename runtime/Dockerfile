# OpenManus Platform — production image (FastAPI API + web UI + Manus agent runtime).
#
# Build context is this directory (runtime/). Referenced by ../render.yaml.
#
#   docker build -t openmanus-platform .
#   docker run -p 8000:8000 -v openmanus-data:/app/OpenManus/workspace \
#       -e LLM_MODEL=... -e LLM_BASE_URL=... -e LLM_API_KEY=... openmanus-platform
#
# Pinned to bookworm: playwright 1.51's `install --with-deps` targets Debian 12.
FROM python:3.12-slim-bookworm

# PLAYWRIGHT_BROWSERS_PATH  Chromium lives at a fixed, user-independent path.
# TIKTOKEN_CACHE_DIR        Encodings are pre-baked below so the first task needs no download.
# HOST / PORT               Bind on all interfaces; hosting platforms such as Render override PORT.
# GIT_*                     Identity used when the platform commits inside project workspaces
#                           (git refuses to commit without one). Override per deployment.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    HOST=0.0.0.0 \
    PORT=8000 \
    GIT_AUTHOR_NAME="OpenManus Platform" \
    GIT_AUTHOR_EMAIL="openmanus@users.noreply.github.com" \
    GIT_COMMITTER_NAME="OpenManus Platform" \
    GIT_COMMITTER_EMAIL="openmanus@users.noreply.github.com"

WORKDIR /app/OpenManus

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so this (slow) layer is cached across source-only changes.
# requirements-platform.txt is the lean runtime subset of requirements.txt; the
# full file drags in browsergym -> torch + CUDA (~4.6 GB) which the server never uses.
# `playwright install --with-deps` runs its own apt-get update for Chromium's libraries.
# app/llm.py uses the model's tiktoken encoding and falls back to cl100k_base.
COPY requirements-platform.txt ./
RUN pip install -r requirements-platform.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"

COPY . .

# SQLite database + per-project git checkouts live here. Mount a volume or a
# persistent disk at this exact path to keep data across deploys/restarts.
RUN mkdir -p workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# The entrypoint writes config/config.toml from LLM_* env vars (or a mounted
# secret file) and then execs the command below.
ENTRYPOINT ["python", "deploy/entrypoint.py"]
CMD ["python", "run_platform.py"]
