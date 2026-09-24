"""Optional HTTP Basic authentication for the platform.

The platform can run agent tasks that spend LLM credits and push to GitHub, so a
publicly reachable deployment must not be left open. Setting ``PLATFORM_PASSWORD``
(and optionally ``PLATFORM_USERNAME``, default ``admin``) protects every route
except the health check, which hosting platforms poll without credentials.

HTTP Basic is used deliberately: the web UI only makes same-origin requests, so
after the browser's one-time prompt it transparently attaches credentials to
``fetch`` and ``EventSource`` calls alike. Always deploy behind HTTPS (Render
terminates TLS for you).
"""

from __future__ import annotations

import base64
import secrets
from typing import Iterable

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send


class BasicAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        username: str,
        password: str,
        exempt_paths: Iterable[str] = ("/api/health",),
        realm: str = "OpenManus Platform",
    ) -> None:
        if not password:
            raise ValueError("BasicAuthMiddleware requires a non-empty password")
        self.app = app
        self._expected = f"{username}:{password}".encode()
        self.exempt_paths = frozenset(exempt_paths)
        self.realm = realm

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                scheme, _, credentials = value.partition(b" ")
                if scheme.lower() != b"basic":
                    return False
                try:
                    decoded = base64.b64decode(credentials.strip(), validate=True)
                except (ValueError, TypeError):
                    return False
                return secrets.compare_digest(decoded, self._expected)
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt_paths or self._authorized(scope):
            await self.app(scope, receive, send)
            return

        response = Response(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": f'Basic realm="{self.realm}", charset="UTF-8"'},
        )
        await response(scope, receive, send)
