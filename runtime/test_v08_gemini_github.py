"""v0.8 — Gemini 3 compatibility and two-token, hardened GitHub access."""

from __future__ import annotations

import asyncio
import os
import stat
import subprocess
from pathlib import Path

import httpx
import pytest

from app.llm import LLM, is_gemini3_model, temperature_params
from app.platform.credentials import github_tokens, is_git_auth_failure
from app.platform.git import (
    ASKPASS_SCRIPT,
    GitError,
    GitWorkspace,
    credential_free_env,
    risky_config_entries,
)
from app.platform.github import GitHubClient
from app.schema import Message, ToolCall


def run(coro):
    return asyncio.run(coro)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(autouse=True)
def _no_ambient_tokens(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_CLASSIC_TOKEN", raising=False)


# ------------------------------------------------------------------ Gemini

SIG = {"google": {"thought_signature": "c2lnbmF0dXJl"}}


def test_thought_signature_survives_round_trip():
    from openai.types.chat import ChatCompletionMessageToolCall

    sdk_call = ChatCompletionMessageToolCall.model_validate(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
            "extra_content": SIG,
        }
    )
    msg = Message.from_tool_calls(tool_calls=[sdk_call], content="")
    assert msg.tool_calls[0].extra_content == SIG
    sent = LLM.format_messages([msg])
    assert sent[0]["tool_calls"][0]["extra_content"] == SIG
    assert sent[0]["tool_calls"][0]["function"]["name"] == "bash"


def test_tool_calls_without_extra_content_have_no_extra_key():
    call = ToolCall(id="c", function={"name": "x", "arguments": "{}"})
    msg = Message.from_tool_calls(tool_calls=[call])
    assert "extra_content" not in msg.to_dict()["tool_calls"][0]


@pytest.mark.parametrize(
    "model,temp,expected",
    [
        ("gemini-3-pro-preview", 0.0, {}),
        ("google/gemini-3-flash", 0.2, {}),
        ("gemini-3-pro-preview", 1.0, {"temperature": 1.0}),
        ("gemini-3-pro-preview", 1.3, {"temperature": 1.3}),
        ("gemini-2.5-pro", 0.0, {"temperature": 0.0}),
        ("claude-sonnet-4-5", 0.0, {"temperature": 0.0}),
        ("gemini-3-pro-preview", None, {}),
    ],
)
def test_temperature_not_lowered_for_gemini3(model, temp, expected):
    assert temperature_params(model, temp) == expected


def test_gemini3_detection():
    assert is_gemini3_model("models/gemini-3-pro")
    assert not is_gemini3_model("gemini-2.5-flash")


# ---------------------------------------------------------- token selection


def test_token_order_and_dedup(monkeypatch):
    assert github_tokens() == []
    monkeypatch.setenv("GITHUB_CLASSIC_TOKEN", "classic")
    assert github_tokens() == ["classic"]
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    assert github_tokens() == ["fine", "classic"]
    monkeypatch.setenv("GITHUB_CLASSIC_TOKEN", "fine")
    assert github_tokens() == ["fine"]


def _mock_github(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr("app.platform.github.httpx.AsyncClient", factory)


@pytest.mark.parametrize("refusal", [401, 403, 404])
def test_api_falls_back_to_classic_token_when_refused(monkeypatch, refusal):
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    monkeypatch.setenv("GITHUB_CLASSIC_TOKEN", "classic")
    seen = []

    def handler(request):
        auth = request.headers["Authorization"]
        seen.append(auth)
        if auth == "Bearer fine":
            return httpx.Response(refusal, json={"message": "nope"})
        return httpx.Response(200, json={"login": "me"})

    _mock_github(monkeypatch, handler)
    assert run(GitHubClient().get_user())["login"] == "me"
    assert seen == ["Bearer fine", "Bearer classic"]


def test_api_uses_only_primary_when_it_works_or_error_is_not_auth(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    monkeypatch.setenv("GITHUB_CLASSIC_TOKEN", "classic")
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "me"})
        return httpx.Response(422, json={"message": "Validation Failed"})

    _mock_github(monkeypatch, handler)
    run(GitHubClient().get_user())
    with pytest.raises(Exception, match="422"):
        run(GitHubClient().create_repository("x"))
    assert seen == ["Bearer fine", "Bearer fine"]


def test_api_without_classic_token_reports_refusal(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    _mock_github(monkeypatch, lambda r: httpx.Response(401, json={"message": "Bad credentials"}))
    with pytest.raises(httpx.HTTPStatusError):
        run(GitHubClient().get_user())


# ------------------------------------------------------------- git: askpass


def _askpass(tmp_path, prompt, token="s3cret"):
    script = tmp_path / "askpass"
    script.write_text(ASKPASS_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return subprocess.run(
        [str(script), prompt], capture_output=True, text=True,
        env={"OPENMANUS_GIT_TOKEN": token, "PATH": os.environ["PATH"]},
    )


def test_askpass_answers_only_github(tmp_path):
    user = _askpass(tmp_path, "Username for 'https://github.com': ")
    assert user.returncode == 0 and user.stdout.strip() == "x-access-token"
    pw = _askpass(tmp_path, "Password for 'https://x-access-token@github.com': ")
    assert pw.returncode == 0 and pw.stdout.strip() == "s3cret"
    for prompt in (
        "Username for 'https://evil.example': ",
        "Password for 'https://x-access-token@evil.example': ",
        "Password for 'https://x-access-token@github.com.evil.example': ",
        "Username for 'https://github.com.evil.example': ",
    ):
        res = _askpass(tmp_path, prompt)
        assert res.returncode != 0 and "s3cret" not in res.stdout, prompt


# ---------------------------------------------------------- git: environment


def test_credential_free_env_strips_secrets_and_git_injection():
    env = credential_free_env({
        "PATH": "/bin", "HOME": "/root",
        "GITHUB_TOKEN": "a", "GITHUB_CLASSIC_TOKEN": "b", "LLM_API_KEY": "c",
        "GIT_ASKPASS": "/x", "SSH_ASKPASS": "/y", "GIT_SSH_COMMAND": "evil",
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_PARAMETERS": "'credential.helper'='!evil'",
        "OPENMANUS_GIT_TOKEN": "d",
    })
    for key in ("GITHUB_TOKEN", "GITHUB_CLASSIC_TOKEN", "LLM_API_KEY", "GIT_ASKPASS",
                "SSH_ASKPASS", "GIT_SSH_COMMAND", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0",
                "GIT_CONFIG_PARAMETERS", "OPENMANUS_GIT_TOKEN"):
        assert key not in env
    assert env["PATH"] == "/bin"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull


def test_local_git_commands_never_get_a_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    captured = []
    ws = GitWorkspace(tmp_path / "p")
    real_spawn = ws._spawn

    async def spy(args, cwd, token):
        captured.append(token)
        return await real_spawn(args, cwd, token)

    ws._spawn = spy
    run(ws.init())
    run(ws.status())
    assert captured and all(t is None for t in captured)


# ------------------------------------------------------ git: auth fallback


def test_git_auth_failure_detection():
    assert is_git_auth_failure("fatal: Authentication failed for 'https://github.com/a/b.git/'")
    assert is_git_auth_failure("remote: Permission to a/b.git denied to bot.\nfatal: ... 403")
    assert is_git_auth_failure("remote: Repository not found.")
    assert not is_git_auth_failure("! [rejected] main -> main (non-fast-forward)")


def _scripted_workspace(tmp_path, monkeypatch, results):
    monkeypatch.setenv("GITHUB_TOKEN", "fine")
    monkeypatch.setenv("GITHUB_CLASSIC_TOKEN", "classic")
    ws = GitWorkspace(tmp_path / "p")
    calls = []

    async def fake(args, cwd, token):
        calls.append((token, args))
        return results.pop(0)

    ws._spawn = fake
    return ws, calls


def test_git_retries_with_classic_token_on_auth_error(tmp_path, monkeypatch):
    ws, calls = _scripted_workspace(tmp_path, monkeypatch, [
        (128, "", "remote: Permission to me/repo.git denied\nfatal: unable to access: The requested URL returned error: 403"),
        (0, "", "To https://github.com/me/repo.git\n * [new branch] x -> x"),
    ])
    out = run(ws._exec_authenticated(["push", "origin", "x"], ws.path))
    assert "new branch" in out
    assert [c[0] for c in calls] == ["fine", "classic"]
    args = calls[0][1]
    assert "credential.helper=" in args and "core.hooksPath=/dev/null" in args


def test_git_does_not_retry_on_non_auth_error(tmp_path, monkeypatch):
    ws, calls = _scripted_workspace(tmp_path, monkeypatch, [
        (1, "", "! [rejected] x -> x (non-fast-forward)"),
    ])
    with pytest.raises(GitError, match="non-fast-forward"):
        run(ws._exec_authenticated(["push", "origin", "x"], ws.path))
    assert [c[0] for c in calls] == ["fine"]


def test_git_error_output_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fine-token-123")
    ws = GitWorkspace(tmp_path / "p")
    code, out, err = run(ws._spawn(["--version"], ws.path, None))
    assert code == 0
    from app.platform.credentials import redact

    assert redact("x fine-token-123 y", ws.tokens) == "x *** y"


# --------------------------------------------------------- git: push safety


@pytest.fixture
def repo_with_remote(tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    ws = GitWorkspace(tmp_path / "proj")
    run(ws.init())
    (ws.path / "a.txt").write_text("hi")
    run(ws.commit("first"))
    run(ws.set_remote(str(remote)))
    return ws, remote


def test_push_skips_hooks(repo_with_remote):
    ws, remote = repo_with_remote
    hook = ws.path / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\ntouch HOOK_RAN\nexit 1\n")
    hook.chmod(0o755)
    run(ws.push())
    assert git("ls-tree", "--name-only", "main", cwd=remote) == "a.txt"
    assert not (ws.path / "HOOK_RAN").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("credential.helper", "!sh -c 'cat > /tmp/stolen'"),
        ("credential.https://github.com.helper", "store"),
        ("core.hooksPath", "/tmp/hooks"),
        ("core.sshCommand", "evil"),
        ("core.fsmonitor", "evil"),
        ("url.https://evil.example/.insteadOf", "https://github.com/"),
        ("remote.origin.pushurl", "https://evil.example/x.git"),
        ("include.path", "/tmp/other.cfg"),
        ("http.extraHeader", "X: y"),
        ("remote.origin.receivepack", "evil"),
    ],
)
def test_push_refused_for_risky_config(repo_with_remote, key, value):
    ws, remote = repo_with_remote
    git("config", "--local", key, value, cwd=ws.path)
    with pytest.raises(GitError, match="Refusing to push"):
        run(ws.push())
    with pytest.raises(subprocess.CalledProcessError):
        git("rev-parse", "main", cwd=remote)


def test_push_refused_for_transport_helper_remote(repo_with_remote):
    ws, _ = repo_with_remote
    git("remote", "set-url", "origin", "ext::sh -c evil", cwd=ws.path)
    with pytest.raises(GitError, match="Refusing to push"):
        run(ws.push())


def test_risky_config_entries_ignores_benign_settings():
    listing = "\n".join([
        "core.repositoryformatversion=0", "core.bare=false", "user.name=x",
        "remote.origin.url=https://github.com/a/b.git",
        "remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*",
        "branch.main.remote=origin",
    ])
    assert risky_config_entries(listing) == []
