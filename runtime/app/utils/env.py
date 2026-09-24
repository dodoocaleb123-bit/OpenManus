"""Control which environment variables agent-executed code can see.

The agent runs arbitrary shell commands, Python snippets and project test
suites. None of those should be able to read the platform's own credentials
(LLM key, GitHub token, admin password): a prompt-injected web page could
otherwise ask the agent to print them. Git/GitHub operations that need the
token go through the platform's own plumbing (``app.platform.git`` /
``app.platform.github``), never through the agent's shell.
"""

from __future__ import annotations

import os
import re
from typing import Mapping

# Exact names that are always removed.
SENSITIVE_ENV_VARS: tuple[str, ...] = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GIT_ASKPASS",
    "LLM_API_KEY",
    "LLM_VISION_API_KEY",
    "PLATFORM_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "BROWSER_USE_API_KEY",
    "DAYTONA_API_KEY",
)

# Anything that *looks* like a credential is removed too.
_SENSITIVE_PATTERN = re.compile(
    r"(API_KEY|APIKEY|SECRET|PASSWORD|PASSWD|TOKEN|PRIVATE_KEY|CREDENTIAL)",
    re.IGNORECASE,
)


def is_sensitive(name: str) -> bool:
    return name in SENSITIVE_ENV_VARS or bool(_SENSITIVE_PATTERN.search(name))


def scrubbed_env(
    base: Mapping[str, str] | None = None, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return a copy of ``base`` (default ``os.environ``) without credentials."""
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if not is_sensitive(k)}
    if extra:
        env.update(extra)
    return env
