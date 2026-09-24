from __future__ import annotations

import asyncio
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.platform.credentials import github_tokens, is_git_auth_failure, redact


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


# Answers git's credential prompts for https://github.com only. Any other host
# (a redirected remote, an insteadOf rewrite, a proxy) gets nothing. The token
# is read from the environment of this one subprocess, never from argv/disk.
ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
  "Username for 'https://github.com'"*) printf '%s\\n' x-access-token ;;
  "Password for 'https://x-access-token@github.com'"*) printf '%s\\n' "$OPENMANUS_GIT_TOKEN" ;;
  *) exit 1 ;;
esac
"""

# Passed as `git -c` for every network operation that may carry a token: no
# credential helpers (they could store/leak the token or answer for us), no
# hooks, and only sane transports.
AUTH_CONFIG_OVERRIDES: tuple[str, ...] = (
    "credential.helper=",
    "core.hooksPath=/dev/null",
    "core.askPass=",
    "core.sshCommand=",
    "core.fsmonitor=false",
    "protocol.allow=never",
    "protocol.https.allow=always",
    "protocol.file.allow=always",
    "http.extraHeader=",
)

# Environment variables that must never reach a git subprocess from the
# platform process: they can inject config, credentials, or helper programs.
_GIT_ENV_BLOCKLIST_PREFIXES = ("GIT_CONFIG", "GCM_", "GIT_TRACE")
_GIT_ENV_BLOCKLIST = frozenset({
    "GIT_ASKPASS", "SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "GIT_SSH", "GIT_SSH_COMMAND",
    "GIT_PROXY_COMMAND", "GIT_EXEC_PATH", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES", "GIT_EXTERNAL_DIFF", "GIT_PAGER", "GIT_EDITOR",
    "GIT_SEQUENCE_EDITOR", "GIT_ALLOW_PROTOCOL", "GIT_PROTOCOL_FROM_USER",
    "OPENMANUS_GIT_TOKEN",
})

# Local config keys that make an authenticated push unsafe (they could run
# programs, redirect the push, or siphon credentials). Matched on the
# lower-cased key; `*` spans subsections.
_RISKY_CONFIG_PATTERNS = tuple(re.compile(p) for p in (
    r"credential\..*",
    r"core\.(hookspath|sshcommand|askpass|fsmonitor|gitproxy)",
    r"include\.path", r"includeif\..*",
    r"url\..*\.(insteadof|pushinsteadof)",
    r"remote\..*\.(pushurl|receivepack|uploadpack|proxy|vcs)",
    r"http\..*",
    r"protocol\..*",
    r"extensions\.worktreeconfig",
    r"(push|remote\..*)\.signingkey", r"gpg\..*",
))


def risky_config_entries(config_text: str) -> list[str]:
    """Risky `key=value` entries from `git config --list` output (values not echoed)."""
    risky = []
    for line in config_text.splitlines():
        key, _, value = line.partition("=")
        lowered = key.strip().lower()
        if any(p.fullmatch(lowered) for p in _RISKY_CONFIG_PATTERNS):
            risky.append(key.strip())
        elif re.fullmatch(r"remote\..*\.url", lowered) and (
            "::" in value or value.strip().startswith(("ext:", "fd:", "-"))
        ):
            risky.append(key.strip())
    return risky


def credential_free_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for git: no platform secrets, no injected git config/helpers."""
    from app.utils.env import scrubbed_env

    env = scrubbed_env(base)
    for key in list(env):
        if key in _GIT_ENV_BLOCKLIST or key.startswith(_GIT_ENV_BLOCKLIST_PREFIXES):
            del env[key]
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GCM_INTERACTIVE"] = "never"
    return env


class GitWorkspace:
    def __init__(
        self,
        workspace: str | Path,
        token: str | None = None,
        fallback_token: str | None = None,
    ):
        self.tokens = github_tokens(token, fallback_token)
        self.path = Path(workspace).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def token(self) -> str | None:
        return self.tokens[0] if self.tokens else None

    @contextmanager
    def _git_env(self, token: str | None = None) -> Iterator[dict[str, str]]:
        """Credential-free git environment; with ``token``, adds a github.com-only askpass.

        The token never appears in argv, remote URLs or .git/config, and is only
        present in the environment of authenticated network commands.
        """
        env = credential_free_env()
        askpass = None
        if token:
            fd, askpass = tempfile.mkstemp(suffix="-git-askpass")
            with os.fdopen(fd, "w") as fh:
                fh.write(ASKPASS_SCRIPT)
            os.chmod(askpass, 0o700)
            env["GIT_ASKPASS"] = askpass
            env["OPENMANUS_GIT_TOKEN"] = token
        try:
            yield env
        finally:
            if askpass:
                try:
                    os.unlink(askpass)
                except FileNotFoundError:
                    pass

    async def _spawn(self, args: list[str], cwd: Path, token: str | None) -> tuple[int, str, str]:
        with self._git_env(token) as env:
            proc = await asyncio.create_subprocess_exec(
                "git", *args, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
        return (
            proc.returncode,
            redact(out.decode(errors="replace"), self.tokens),
            redact(err.decode(errors="replace"), self.tokens),
        )

    async def _exec(self, args: list[str], cwd: Path, check: bool = True) -> str:
        """Local git command: never gets a token."""
        code, stdout, stderr = await self._spawn(args, cwd, None)
        if check and code != 0:
            raise GitError(stderr.strip() or stdout.strip() or f"git exited {code}")
        return stdout.strip()

    async def _exec_authenticated(self, args: list[str], cwd: Path) -> str:
        """Network git command: GITHUB_TOKEN first, GITHUB_CLASSIC_TOKEN on auth failure.

        Returns stdout+stderr (git reports push/clone progress on stderr).
        """
        overrides = [item for pair in (("-c", o) for o in AUTH_CONFIG_OVERRIDES) for item in pair]
        attempts: list[str | None] = list(self.tokens) or [None]
        text, code = "", 1
        for index, token in enumerate(attempts):
            code, stdout, stderr = await self._spawn(overrides + args, cwd, token)
            text = (stdout + stderr).strip()
            if code == 0:
                return text
            if index + 1 < len(attempts) and is_git_auth_failure(text):
                continue
            break
        raise GitError(text or f"git exited {code}")

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
        args += ["--", remote, str(self.path)]
        await self._exec_authenticated(args, parent)
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
        await self._assert_safe_for_push()
        text = await self._exec_authenticated(
            ["push", "--no-verify", "-u", "origin", branch], self.path
        )
        return text or f"Pushed {branch}"

    async def _assert_safe_for_push(self) -> None:
        """Refuse to hand credentials to git if .git/config was tampered with.

        The agent's shell can edit the workspace's .git/config; settings such as
        credential helpers, hooksPath, insteadOf rewrites or pushurl could run
        programs or send the token elsewhere.
        """
        config = self.path / ".git" / "config"
        if not config.is_file():
            raise GitError("Refusing to push: .git/config is missing or not a regular file")
        listing = await self._exec(
            ["config", "--file", str(config), "--no-includes", "--list"], self.path, check=False
        )
        risky = risky_config_entries(listing)
        if risky:
            raise GitError(
                "Refusing to push: .git/config contains risky settings ("
                + ", ".join(sorted(set(risky)))
                + "). Remove them (e.g. `git config --unset <key>`) and try again."
            )

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
