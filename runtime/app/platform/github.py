from __future__ import annotations

import os
from typing import Any

import httpx


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
