"""GitHub credential selection shared by the API client and git plumbing.

Two tokens are supported:

* ``GITHUB_TOKEN``         - primary (typically fine-grained), always tried first.
* ``GITHUB_CLASSIC_TOKEN`` - optional fallback (typically classic ``repo`` scope),
  used only when GitHub refuses the primary (API 401/403/404, git auth errors).
"""

from __future__ import annotations

import os
import re

PRIMARY_ENV = "GITHUB_TOKEN"
FALLBACK_ENV = "GITHUB_CLASSIC_TOKEN"

# HTTP statuses that mean "this token cannot do that" on the GitHub API. GitHub
# answers 404 instead of 403 for private resources the token cannot see.
REFUSED_STATUSES = frozenset({401, 403, 404})

_GIT_AUTH_FAILURE_RE = re.compile(
    r"authentication failed"
    r"|could not read (username|password)"
    r"|invalid username or (password|token)"
    r"|terminal prompts disabled"
    r"|permission to .* denied"
    r"|write access to repository not granted"
    r"|repository not found"
    r"|requested url returned error: 40[134]"
    r"|the requested url returned error: 40[134]"
    r"|http 40[134]"
    r"|403 forbidden",
    re.IGNORECASE,
)


def github_tokens(primary: str | None = None, fallback: str | None = None) -> list[str]:
    """Distinct non-empty tokens in the order they should be tried."""
    tokens: list[str] = []
    for token in (
        primary or os.getenv(PRIMARY_ENV),
        fallback or os.getenv(FALLBACK_ENV),
    ):
        token = (token or "").strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def any_token_configured() -> bool:
    return bool(github_tokens())


def is_git_auth_failure(output: str) -> bool:
    return bool(_GIT_AUTH_FAILURE_RE.search(output or ""))


def redact(text: str, tokens: list[str]) -> str:
    for token in tokens:
        if token:
            text = text.replace(token, "***")
    return text
