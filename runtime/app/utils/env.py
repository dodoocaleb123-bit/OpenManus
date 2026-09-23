"""Helpers for controlling which environment variables agent-executed code sees."""

from __future__ import annotations

import os

# Platform credentials that must never be visible to code the agent runs
# (workspace test scripts, python_execute, bash sessions). The GitHub token is
# kept only inside the platform's own git/API plumbing.
SENSITIVE_ENV_VARS: tuple[str, ...] = ("GITHUB_TOKEN", "GH_TOKEN")


def scrubbed_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment without platform credentials."""
    env = dict(os.environ if base is None else base)
    for var in SENSITIVE_ENV_VARS:
        env.pop(var, None)
    return env
