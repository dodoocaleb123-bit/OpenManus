from __future__ import annotations

import asyncio
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class GitError(RuntimeError):
    pass


# Never committed from a project workspace, whatever the project's .gitignore
# says. Written to .git/info/exclude (local only, not part of the repo).
LOCAL_EXCLUDES = (
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".next/",
    ".turbo/",
    ".env",
    ".env.local",
    ".openmanus/",
    "*.log",
)

_BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]{1,100}")
_GITHUB_REMOTE_RE = re.compile(r"github\.com[:/]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def valid_branch_name(name: str) -> bool:
    return bool(
        _BRANCH_RE.fullmatch(name)
        and not name.startswith(("/", "-", "."))
        and not name.endswith(("/", ".", ".lock"))
        and ".." not in name
        and "//" not in name
    )


class GitWorkspace:
    def __init__(self, workspace: str | Path, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.path = Path(workspace).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _git_env(self) -> Iterator[dict[str, str]]:
        """Environment for git subprocesses; supplies the token via GIT_ASKPASS.

        The token never appears in argv, remote URLs or .git/config.
        """
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        askpass = None
        if self.token:
            askpass = tempfile.NamedTemporaryFile("w", delete=False, suffix="-git-askpass")
            askpass.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  Username*) printf '%s\\n' x-access-token ;;\n"
                "  *) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
                "esac\n"
            )
            askpass.close()
            os.chmod(askpass.name, 0o700)
            env["GIT_ASKPASS"] = askpass.name
            env["GITHUB_TOKEN"] = self.token
        try:
            yield env
        finally:
            if askpass:
                try:
                    os.unlink(askpass.name)
                except FileNotFoundError:
                    pass

    async def _exec(self, args: list[str], cwd: Path, check: bool = True) -> str:
        with self._git_env() as env:
            proc = await asyncio.create_subprocess_exec(
                "git", *args, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
        stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
        if check and proc.returncode != 0:
            message = stderr.strip() or stdout.strip() or f"git exited {proc.returncode}"
            if self.token:
                message = message.replace(self.token, "***")
            raise GitError(message)
        return stdout.strip()

    async def _run(self, *args: str, check: bool = True) -> str:
        return await self._exec(list(args), self.path, check)

    # ------------------------------------------------------------------ state

    async def is_repo(self) -> bool:
        """True only if the workspace itself is the root of a git repository.

        A project folder that merely sits *inside* another repository (e.g. the
        platform's own checkout during local development) does not count.
        """
        top = await self._run("rev-parse", "--show-toplevel", check=False)
        return bool(top) and Path(top).resolve() == self.path

    async def _require_own_repo(self) -> None:
        if not await self.is_repo():
            raise GitError(
                "This project has no git repository yet. Connect a GitHub repository, "
                "or initialise one (the agent's `platform_git` tool can do this with action `init`)."
            )

    async def has_commits(self) -> bool:
        return bool(await self._run("rev-parse", "--verify", "-q", "HEAD", check=False))

    async def ensure_local_excludes(self) -> None:
        exclude = self.path / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8").splitlines() if exclude.exists() else []
        missing = [p for p in LOCAL_EXCLUDES if p not in existing]
        if missing:
            with exclude.open("a", encoding="utf-8") as fh:
                if existing and existing[-1].strip():
                    fh.write("\n")
                fh.write("# added by OpenManus Platform\n" + "\n".join(missing) + "\n")

    async def _identity_args(self) -> list[str]:
        """Fallback author identity when neither env nor git config provide one."""
        if os.environ.get("GIT_AUTHOR_NAME") and os.environ.get("GIT_AUTHOR_EMAIL"):
            return []
        name = await self._run("config", "user.name", check=False)
        email = await self._run("config", "user.email", check=False)
        args: list[str] = []
        if not name:
            args += ["-c", "user.name=OpenManus Agent"]
        if not email:
            args += ["-c", "user.email=agent@openmanus.local"]
        return args

    # ------------------------------------------------------------- operations

    async def init(self, default_branch: str = "main") -> str:
        if await self.is_repo():
            return await self.current_branch()
        if not valid_branch_name(default_branch):
            raise GitError("Invalid branch name")
        await self._run("init", "-b", default_branch)
        await self.ensure_local_excludes()
        return default_branch

    async def clone(self, remote: str, branch: str | None = None) -> None:
        if any(self.path.iterdir()):
            raise GitError("Project workspace is not empty")
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        args = ["clone"]
        if branch:
            args += ["--branch", branch]
        args += [remote, str(self.path)]
        await self._exec(args, parent)
        await self.ensure_local_excludes()

    async def current_branch(self) -> str:
        branch = await self._run("branch", "--show-current", check=False)
        if branch:
            return branch
        # Fresh repository without commits: HEAD points at an unborn branch.
        ref = await self._run("symbolic-ref", "--short", "HEAD", check=False)
        return ref

    async def create_branch(self, name: str) -> str:
        await self._require_own_repo()
        if not valid_branch_name(name):
            raise GitError("Invalid branch name")
        await self._run("switch", "-c", name)
        return await self.current_branch()

    async def checkout(self, name: str) -> str:
        await self._require_own_repo()
        if not valid_branch_name(name):
            raise GitError("Invalid branch name")
        await self._run("switch", name)
        return await self.current_branch()

    async def status(self) -> dict:
        await self._require_own_repo()
        branch = await self.current_branch()
        porcelain = await self._run("status", "--porcelain=v1", "-b")
        diff = await self._run("diff", "--stat", check=False)
        return {"branch": branch, "status": porcelain, "diff_stat": diff}

    async def diff(self) -> str:
        await self._require_own_repo()
        # Include untracked files so brand-new files show up in review too.
        await self._run("add", "-A", "--intent-to-add", check=False)
        return await self._run("diff", check=False)

    async def log(self, limit: int = 10) -> str:
        await self._require_own_repo()
        if not await self.has_commits():
            return ""
        return await self._run("log", f"-{max(1, min(limit, 100))}", "--oneline", "--decorate")

    async def commit(self, message: str) -> str:
        await self._require_own_repo()
        message = message.strip()
        if not message:
            raise GitError("Commit message is required")
        await self.ensure_local_excludes()
        await self._run("add", "-A")
        if not await self._run("status", "--porcelain"):
            raise GitError("Nothing to commit: the working tree is clean")
        identity = await self._identity_args()
        return await self._run(*identity, "commit", "-m", message)

    async def set_remote(self, url: str, name: str = "origin") -> None:
        await self._require_own_repo()
        remotes = (await self._run("remote", check=False)).split()
        if name in remotes:
            await self._run("remote", "set-url", name, url)
        else:
            await self._run("remote", "add", name, url)

    async def push(self, branch: str | None = None) -> str:
        await self._require_own_repo()
        branch = branch or await self.current_branch()
        if not await self.has_commits():
            raise GitError("Nothing to push: make a commit first")
        # git writes push progress to stderr; return both so callers see the result.
        with self._git_env() as env:
            proc = await asyncio.create_subprocess_exec(
                "git", "push", "-u", "origin", branch, cwd=self.path, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
        text = (out.decode(errors="replace") + err.decode(errors="replace")).strip()
        if self.token:
            text = text.replace(self.token, "***")
        if proc.returncode != 0:
            raise GitError(text or f"git push exited {proc.returncode}")
        return text or f"Pushed {branch}"

    async def remote_url(self) -> str:
        await self._require_own_repo()
        return await self._run("remote", "get-url", "origin")

    async def github_repository(self) -> tuple[str, str] | None:
        """(owner, repo) parsed from the origin remote, if it points at GitHub."""
        url = await self._run("remote", "get-url", "origin", check=False)
        match = _GITHUB_REMOTE_RE.search(url or "")
        return (match.group(1), match.group(2)) if match else None


def github_remote(owner: str, repo: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub repository")
    return f"https://github.com/{owner}/{repo}.git"
