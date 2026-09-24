"""Project-level git/GitHub workflow shared by the HTTP API and the agent tool.

Keeps the project's stored state (repository, branch, git_ready) in sync with
what is actually on disk, so both humans (UI) and the agent (platform_git
tool) see the same thing after a reload or a restart.
"""

from __future__ import annotations

import re
from typing import Any

from app.platform.git import GitError, GitWorkspace, github_remote
from app.platform.github import GitHubClient, GitHubError
from app.platform.models import Project
from app.platform.store import PlatformStore

_REPO_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}")


class ProjectGit:
    def __init__(self, store: PlatformStore, project: Project, token: str | None = None):
        self.store = store
        self.project = project
        self.token = token
        self.git = GitWorkspace(project.workspace, token=token)

    def _github(self) -> GitHubClient:
        return GitHubClient(self.token)

    async def sync(self) -> Project:
        """Refresh repository/branch from disk and persist them."""
        if await self.git.is_repo():
            branch = await self.git.current_branch()
            remote = await self.git.github_repository()
            self.project.branch = branch or self.project.branch
            if remote:
                self.project.repository = f"{remote[0]}/{remote[1]}"
            self.project.git_ready = True
        else:
            self.project.git_ready = False
        self.store.update_project(self.project)
        return self.project

    async def connect(self, owner: str, repo: str, branch: str | None = None) -> dict[str, Any]:
        remote = github_remote(owner, repo)
        await self.git.clone(remote, branch)
        await self.sync()
        return {"project": self.project, "git": await self.git.status()}

    async def init(self, default_branch: str = "main") -> Project:
        await self.git.init(default_branch)
        return await self.sync()

    async def create_branch(self, name: str) -> str:
        branch = await self.git.create_branch(name)
        await self.sync()
        return branch

    async def checkout(self, name: str) -> str:
        branch = await self.git.checkout(name)
        await self.sync()
        return branch

    async def commit(self, message: str) -> str:
        result = await self.git.commit(message)
        await self.sync()
        return result

    async def push(self) -> str:
        return await self.git.push(await self.git.current_branch() or self.project.branch)

    async def pull_request(
        self, title: str, body: str = "", base: str | None = None, draft: bool = False
    ) -> dict[str, Any]:
        remote = await self.git.github_repository()
        if not remote:
            raise GitError("This project is not connected to a GitHub repository")
        head = await self.git.current_branch()
        if not head:
            raise GitError("Could not determine the current branch")
        # Make sure GitHub has the latest commits before opening the PR.
        await self.git.push(head)
        pr = await self._github().create_pull_request(
            remote[0], remote[1], head=head, title=title.strip() or head, body=body, base=base, draft=draft
        )
        return {
            "number": pr.get("number"),
            "url": pr.get("html_url"),
            "title": pr.get("title"),
            "head": head,
            "base": (pr.get("base") or {}).get("ref", base),
            "already_existed": bool(pr.get("already_existed")),
        }

    async def publish(
        self,
        name: str,
        private: bool = True,
        description: str = "",
        commit_message: str = "Initial commit",
    ) -> dict[str, Any]:
        """Create a new GitHub repository for a local project and push it.

        Used when the agent builds a project from scratch. Initialises git and
        makes the first commit if needed.
        """
        if not _REPO_NAME_RE.fullmatch(name):
            raise GitError("Repository name may contain letters, numbers, '-', '_' and '.'")
        if await self.git.is_repo() and await self.git.github_repository():
            raise GitError(f"Project is already connected to {self.project.repository}")
        await self.git.init("main")
        if await self.git._run("status", "--porcelain") or not await self.git.has_commits():
            try:
                await self.git.commit(commit_message)
            except GitError as exc:
                if "Nothing to commit" not in str(exc):
                    raise
        if not await self.git.has_commits():
            raise GitError("The project is empty; create some files before publishing")
        repo = await self._github().create_repository(name, private=private, description=description)
        owner = (repo.get("owner") or {}).get("login")
        await self.git.set_remote(github_remote(owner, repo["name"]))
        branch = await self.git.current_branch()
        await self.git.push(branch)
        await self.sync()
        return {"repository": repo.get("full_name"), "url": repo.get("html_url"), "private": repo.get("private"), "branch": branch}


__all__ = ["ProjectGit", "GitError", "GitHubError"]
