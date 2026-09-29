from __future__ import annotations

import re
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
        "- connect {owner, repo, branch?}: clone an existing GitHub repository into this project workspace.\n"
        "- status / diff / log: inspect the repository.\n"
        "- init: create a git repository in the workspace (for new projects).\n"
        "- create_branch {branch} / checkout {branch}.\n"
        "- commit {message}: stages ALL changes (node_modules/.venv/.env are always excluded) and commits.\n"
        "- push: push the current branch to GitHub.\n"
        "- create_pull_request {title, body, base?, draft?}: pushes and opens a PR from the current branch.\n"
        "- publish_repository {repo_name, private?, description?}: create a NEW GitHub repository for a "
        "project that has none, commit everything and push. Returns the repository URL. External push/PR/publish "
        "actions require the user to request them explicitly or approve through ask_human."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "connect", "status", "diff", "log", "init", "create_branch", "checkout", "commit", "push",
                    "create_pull_request", "publish_repository",
                ],
            },
            "owner": {"type": "string", "description": "GitHub repository owner for connect"},
            "repo": {"type": "string", "description": "GitHub repository name for connect"},
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
    user_request: str = ""
    request_approval: Any = None  # async (question) -> user's reply

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

    def _explicitly_requested(self, action: str) -> bool:
        text = self.user_request.casefold()
        action_words = {
            "push": r"\bpush\b",
            "create_pull_request": r"\b(?:create|open|submit|draft)\b.{0,40}\b(?:pull request|pr)\b|\b(?:pull request|pr)\b.{0,40}\b(?:create|open|submit)\b",
            "publish_repository": r"\bpublish\b.{0,40}\b(?:github\s+)?(?:repo|repository)\b|\bcreate\b.{0,40}\b(?:github\s+)?(?:repo|repository)\b",
        }
        if not re.search(action_words.get(action, r"$^"), text):
            return False
        if self._explicitly_prohibited(action):
            return False
        return True

    def _explicitly_prohibited(self, action: str) -> bool:
        text = self.user_request.casefold()
        label = {"push": r"push", "create_pull_request": r"(?:pull request|pr)", "publish_repository": r"publish|repository"}.get(action)
        return bool(label and re.search(rf"\b(?:do\s+not|don't|never|avoid|without)\b[^.!?]{{0,80}}\b(?:{label})\b", text))

    async def _approve_external_action(self, action: str, summary: str) -> str | None:
        if self._explicitly_prohibited(action):
            return "The user explicitly prohibited this GitHub action; no remote change was made."
        source = "explicit_user_request" if self._explicitly_requested(action) else "user_approval"
        if source == "user_approval":
            if self.request_approval is None:
                return "This external GitHub action needs explicit user approval before it can proceed."
            answer = await self.request_approval(f"OpenManus requests approval to {summary}. Reply YES to approve, or anything else to cancel.")
            if (answer or "").strip().casefold() not in {"yes", "approve", "approved", "confirm"}:
                return "The user did not approve this external GitHub action; no remote change was made."
        await self._notify("github.action.approved", f"Authorized GitHub action: {action}", {"action": action, "approval_source": source})
        return None

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action")
        try:
            svc = self._service()
            git = svc.git
            external_summaries = {
                "push": "push the current branch and its committed changes to the connected GitHub repository",
                "create_pull_request": "push the current branch and open the requested pull request",
                "publish_repository": "create a GitHub repository and push this project to it",
            }
            if action in external_summaries:
                denied = await self._approve_external_action(action, external_summaries[action])
                if denied:
                    return self.fail_response(denied, evidence={"action": action, "approved": False})
            if action == "connect":
                owner = (kwargs.get("owner") or "").strip()
                repo = (kwargs.get("repo") or "").strip()
                if not owner or not repo:
                    return self.fail_response("connect requires the GitHub owner and repository name")
                result = await svc.connect(owner, repo, kwargs.get("branch") or None)
                await self._notify("github.repository", f"Connected {owner}/{repo}", {"repository": f"{owner}/{repo}", "project": result["project"].model_dump(mode="json")})
                return self.success_response(f"Connected GitHub repository {owner}/{repo} on branch {result['git'].get('branch') or 'default'}", evidence={"action": action, "repository": f"{owner}/{repo}", "branch": result["git"].get("branch")})
            if action == "status":
                result = await git.status()
                return self.success_response(result, evidence={"action": action, **(result if isinstance(result, dict) else {})})
            if action == "diff":
                diff = await git.diff()
                if len(diff) > 20000:
                    diff = diff[:20000] + f"\n…[diff truncated, {len(diff)} chars total]"
                return self.success_response(diff or "No changes.", evidence={"action": action, "changed": bool(diff)})
            if action == "log":
                result = await git.log(15) or "No commits yet."
                return self.success_response(result, evidence={"action": action, "entries": len(result.splitlines())})
            if action == "init":
                project = await svc.init()
                return self.success_response(f"Initialised git repository on branch {project.branch}", evidence={"action": action, "branch": project.branch})
            if action == "create_branch":
                branch = await svc.create_branch(kwargs.get("branch") or "")
                return self.success_response(f"Created and switched to branch {branch}", evidence={"action": action, "branch": branch})
            if action == "checkout":
                branch = await svc.checkout(kwargs.get("branch") or "")
                return self.success_response(f"Switched to branch {branch}", evidence={"action": action, "branch": branch})
            if action == "commit":
                result = await svc.commit(kwargs.get("message") or "")
                return self.success_response(result, evidence={"action": action, "message": kwargs.get("message") or ""})
            if action == "push":
                result = await svc.push()
                return self.success_response(result, evidence={"action": action, "pushed": True})
            if action == "create_pull_request":
                pr = await svc.pull_request(
                    kwargs.get("title") or "",
                    kwargs.get("body") or "",
                    kwargs.get("base") or None,
                    bool(kwargs.get("draft")),
                )
                await self._notify("github.pull_request", f"Pull request #{pr['number']}: {pr['url']}", pr)
                return self.success_response(pr, evidence={"action": action, "number": pr.get("number"), "url": pr.get("url")})
            if action == "publish_repository":
                repo = await svc.publish(
                    kwargs.get("repo_name") or "",
                    private=kwargs.get("private", True) is not False,
                    description=kwargs.get("description") or "",
                )
                await self._notify("github.repository", f"Published {repo['repository']}: {repo['url']}", repo)
                return self.success_response(repo, evidence={"action": action, "repository": repo.get("repository"), "url": repo.get("url")})
            return self.fail_response(f"Unknown action: {action}")
        except (GitError, GitHubError, ValueError) as exc:
            return self.fail_response(str(exc))
        except RuntimeError as exc:  # e.g. GITHUB_TOKEN not configured
            return self.fail_response(str(exc))
