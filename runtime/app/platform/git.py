from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path


class GitError(RuntimeError):
    pass


class GitWorkspace:
    def __init__(self, workspace: str | Path, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.path = Path(workspace).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    async def _run(self, *args: str, check: bool = True) -> str:
        env = os.environ.copy()
        askpass = None
        if self.token:
            askpass = tempfile.NamedTemporaryFile("w", delete=False, suffix="-git-askpass")
            askpass.write("#!/bin/sh\nprintf '%s\n' \"$GITHUB_TOKEN\"\n")
            askpass.close()
            os.chmod(askpass.name, 0o700)
            env["GIT_ASKPASS"] = askpass.name
            env["GITHUB_TOKEN"] = self.token
            env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args, cwd=self.path, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
            if check and proc.returncode != 0:
                raise GitError(stderr.strip() or stdout.strip() or f"git exited {proc.returncode}")
            return stdout.strip()
        finally:
            if askpass:
                try:
                    os.unlink(askpass.name)
                except FileNotFoundError:
                    pass

    async def is_repo(self) -> bool:
        """True only when the workspace itself is the root of a git repository.

        Being *inside* some other repository (e.g. the platform's own checkout)
        must not count: project git operations would otherwise modify it.
        """
        out = await self._run("rev-parse", "--show-toplevel", check=False)
        return bool(out) and Path(out).resolve() == self.path

    async def _require_own_repo(self) -> None:
        if not await self.is_repo():
            raise GitError(
                "Project workspace is not a git repository (connect a GitHub repository first)"
            )

    async def clone(self, remote: str, branch: str | None = None) -> None:
        if any(self.path.iterdir()):
            raise GitError("Project workspace is not empty")
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        args = ["clone"]
        if branch:
            args += ["--branch", branch]
        args += [remote, str(self.path)]
        env = os.environ.copy()
        askpass = None
        if self.token:
            askpass = tempfile.NamedTemporaryFile("w", delete=False, suffix="-git-askpass")
            askpass.write("#!/bin/sh\nprintf '%s\n' \"$GITHUB_TOKEN\"\n")
            askpass.close()
            os.chmod(askpass.name, 0o700)
            env["GIT_ASKPASS"] = askpass.name
            env["GITHUB_TOKEN"] = self.token
            env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args, cwd=parent, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            if proc.returncode != 0:
                raise GitError(err.decode(errors="replace").strip() or out.decode(errors="replace").strip())
        finally:
            if askpass:
                try:
                    os.unlink(askpass.name)
                except FileNotFoundError:
                    pass

    async def current_branch(self) -> str:
        await self._require_own_repo()
        return await self._run("branch", "--show-current")

    async def create_branch(self, name: str) -> str:
        await self._require_own_repo()
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", name) or name.startswith("/") or name.endswith("/"):
            raise GitError("Invalid branch name")
        await self._run("switch", "-c", name)
        return name

    async def status(self) -> dict:
        await self._require_own_repo()
        branch = await self.current_branch()
        porcelain = await self._run("status", "--porcelain=v1", "-b")
        diff = await self._run("diff", "--stat", check=False)
        return {"branch": branch, "status": porcelain, "diff_stat": diff}

    async def diff(self) -> str:
        await self._require_own_repo()
        return await self._run("diff", check=False)

    async def commit(self, message: str) -> str:
        await self._require_own_repo()
        message = message.strip()
        if not message:
            raise GitError("Commit message is required")
        await self._run("add", "-A")
        return await self._run("commit", "-m", message)

    async def push(self, branch: str | None = None) -> str:
        await self._require_own_repo()
        branch = branch or await self.current_branch()
        return await self._run("push", "-u", "origin", branch)

    async def remote_url(self) -> str:
        await self._require_own_repo()
        return await self._run("remote", "get-url", "origin")


def github_remote(owner: str, repo: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub repository")
    return f"https://github.com/{owner}/{repo}.git"
