from __future__ import annotations

import os
from typing import Any

import httpx


class GitHubError(RuntimeError):
    """GitHub API request failed; message is safe to show to users."""


class GitHubClient:
    """Small first-party GitHub API client.

    Uses GITHUB_TOKEN when present. The token is never returned to the agent or API.
    """

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is not configured")
        self.base_url = "https://api.github.com"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        try:
            body = response.json()
            detail = body.get("message", "")
            errors = body.get("errors") or []
            extra = "; ".join(
                e.get("message") or e.get("code", "") for e in errors if isinstance(e, dict)
            )
            if extra:
                detail = f"{detail} ({extra})"
        except Exception:
            detail = response.text[:300]
        raise GitHubError(f"GitHub {action} failed ({response.status_code}): {detail}")

    async def get_user(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/user", headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def list_repositories(self, per_page: int = 50) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{self.base_url}/user/repos",
                headers=self._headers(),
                params={"per_page": min(per_page, 100), "sort": "updated"},
            )
            response.raise_for_status()
            return response.json()

    async def get_repository(self, owner: str, repo: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}", headers=self._headers()
            )
            response.raise_for_status()
            return response.json()

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        head: str,
        title: str,
        body: str = "",
        base: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Open a PR from ``head`` into ``base`` (default: the repo's default branch).

        If a PR for the same head/base is already open, that PR is returned
        instead of failing, so the agent can safely retry.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            if not base:
                info = await client.get(f"{self.base_url}/repos/{owner}/{repo}", headers=self._headers())
                self._raise_for_status(info, "repository lookup")
                base = info.json().get("default_branch") or "main"
            if head == base:
                raise GitHubError(
                    f"Cannot open a pull request from '{head}' into itself; create a feature branch first"
                )
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo}/pulls",
                headers=self._headers(),
                json={"title": title, "head": head, "base": base, "body": body, "draft": draft},
            )
            if response.status_code == 422 and "already exists" in response.text:
                existing = await client.get(
                    f"{self.base_url}/repos/{owner}/{repo}/pulls",
                    headers=self._headers(),
                    params={"head": f"{owner}:{head}", "base": base, "state": "open"},
                )
                self._raise_for_status(existing, "pull request lookup")
                items = existing.json()
                if items:
                    return {**items[0], "already_existed": True}
            self._raise_for_status(response, "pull request creation")
            return response.json()

    async def create_repository(
        self, name: str, private: bool = True, description: str = ""
    ) -> dict[str, Any]:
        """Create an empty repository owned by the authenticated user."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/user/repos",
                headers=self._headers(),
                json={
                    "name": name,
                    "private": private,
                    "description": description[:350],
                    "auto_init": False,
                },
            )
            self._raise_for_status(response, "repository creation")
            return response.json()
