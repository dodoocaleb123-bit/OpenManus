from __future__ import annotations

from typing import Any

from app.platform.git import GitError
from app.platform.git_service import ProjectGit
from app.platform.github import GitHubError
from app.tool.base import BaseTool, ToolResult


class PlatformGitTool(BaseTool):
    """Git + GitHub for the agent, executed by the platform with server-side credentials."""

    name: str = "platform_git"
    description: str = (
        "Version control for this project, executed by the platform with the user's GitHub credentials "
        "(you never see the token). Actions:\n"
        "- status / diff / log: inspect the repository.\n"
        "- init: create a git repository in the workspace (for new projects).\n"
        "- create_branch {branch} / checkout {branch}.\n"
        "- commit {message}: stages ALL changes (node_modules/.venv/.env are always excluded) and commits.\n"
        "- push: push the current branch to GitHub.\n"
        "- create_pull_request {title, body, base?, draft?}: pushes and opens a PR from the current branch.\n"
        "- publish_repository {repo_name, private?, description?}: create a NEW GitHub repository for a "
        "project that has none, commit everything and push. Returns the repository URL."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "diff", "log", "init", "create_branch", "checkout", "commit", "push",
                    "create_pull_request", "publish_repository",
                ],
            },
            "branch": {"type": "string", "description": "Branch name for create_branch/checkout"},
            "message": {"type": "string", "description": "Commit message"},
            "title": {"type": "string", "description": "Pull request title"},
            "body": {"type": "string", "description": "Pull request description (markdown)"},
            "base": {"type": "string", "description": "PR base branch (default: repository default)"},
            "draft": {"type": "boolean", "description": "Open the PR as a draft"},
            "repo_name": {"type": "string", "description": "New repository name for publish_repository"},
            "private": {"type": "boolean", "description": "Make the new repository private (default true)"},
            "description": {"type": "string", "description": "New repository description"},
        },
        "required": ["action"],
    }

    store: Any = None
    project_id: str = ""
    on_event: Any = None  # async (type, message, data) -> None

    class Config:
        arbitrary_types_allowed = True

    def _service(self) -> ProjectGit:
        project = self.store.get_project(self.project_id)
        if project is None:
            raise GitError("Project no longer exists")
        return ProjectGit(self.store, project)

    async def _notify(self, type_: str, message: str, data: dict) -> None:
        if self.on_event is not None:
            await self.on_event(type_, message, data)

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action")
        try:
            svc = self._service()
            git = svc.git
            if action == "status":
                return self.success_response(await git.status())
            if action == "diff":
                diff = await git.diff()
                if len(diff) > 20000:
                    diff = diff[:20000] + f"\n…[diff truncated, {len(diff)} chars total]"
                return self.success_response(diff or "No changes.")
            if action == "log":
                return self.success_response(await git.log(15) or "No commits yet.")
            if action == "init":
                project = await svc.init()
                return self.success_response(f"Initialised git repository on branch {project.branch}")
            if action == "create_branch":
                branch = await svc.create_branch(kwargs.get("branch") or "")
                return self.success_response(f"Created and switched to branch {branch}")
            if action == "checkout":
                branch = await svc.checkout(kwargs.get("branch") or "")
                return self.success_response(f"Switched to branch {branch}")
            if action == "commit":
                return self.success_response(await svc.commit(kwargs.get("message") or ""))
            if action == "push":
                return self.success_response(await svc.push())
            if action == "create_pull_request":
                pr = await svc.pull_request(
                    kwargs.get("title") or "",
                    kwargs.get("body") or "",
                    kwargs.get("base") or None,
                    bool(kwargs.get("draft")),
                )
                await self._notify("github.pull_request", f"Pull request #{pr['number']}: {pr['url']}", pr)
                return self.success_response(pr)
            if action == "publish_repository":
                repo = await svc.publish(
                    kwargs.get("repo_name") or "",
                    private=kwargs.get("private", True) is not False,
                    description=kwargs.get("description") or "",
                )
                await self._notify("github.repository", f"Published {repo['repository']}: {repo['url']}", repo)
                return self.success_response(repo)
            return self.fail_response(f"Unknown action: {action}")
        except (GitError, GitHubError, ValueError) as exc:
            return self.fail_response(str(exc))
        except RuntimeError as exc:  # e.g. GITHUB_TOKEN not configured
            return self.fail_response(str(exc))
