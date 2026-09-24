FROM python:3.12-slim

WORKDIR /app/OpenManus

# git: project repository operations; node/npm: coding-loop validation for Node projects
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv)

COPY . .

RUN uv pip install --system -r requirements.txt

# Chromium for the shared platform browser (see README_PLATFORM.md)
RUN playwright install --with-deps chromium

# Durable data (SQLite store + project workspaces) lives on a mounted volume.
ENV PLATFORM_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

CMD ["python", "run_platform.py"]
