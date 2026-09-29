"""Safety classification for agent shell commands."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandRisk:
    level: str
    reasons: tuple[str, ...]
    requires_confirmation: bool = False


_BLOCKED = (
    (re.compile(r"\bcurl\b[^\n|]*\|\s*(?:ba|z)?sh\b|wget\b[^\n|]*\|\s*(?:ba|z)?sh\b", re.I), "remote script piped to a shell"),
    (re.compile(r"\brm\s+-[^\n]*\b/\s*(?:$|[;&])", re.I), "recursive deletion at filesystem root"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=", re.I), "direct disk modification"),
)
_CONFIRM = (
    (re.compile(r"\b(git\s+push|git\s+reset\s+--hard|git\s+clean\s+-f|force[- ]push)\b", re.I), "destructive or external Git operation"),
    (re.compile(r"\b(rm|rmdir|del|remove-item)\b[^\n]*(?:-r|-rf|-recurse|/s)\b", re.I), "recursive deletion"),
    (re.compile(r"\b(sudo|chmod\s+777|chown\s+|docker\s+(?:rm|system\s+prune))\b", re.I), "privileged or destructive system operation"),
    (re.compile(r"\b(npm|pip|uv|apt(?:-get)?)\s+install\b", re.I), "dependency installation"),
)


def classify_command(command: str) -> CommandRisk:
    text = (command or "").strip()
    if not text or text in {"pwd", "ls", "dir", "whoami", "git status", "git diff", "git log"}:
        return CommandRisk("safe", ())
    blocked = [reason for pattern, reason in _BLOCKED if pattern.search(text)]
    if blocked:
        return CommandRisk("blocked", tuple(blocked), True)
    confirmation = [reason for pattern, reason in _CONFIRM if pattern.search(text)]
    if confirmation:
        return CommandRisk("confirmation", tuple(confirmation), True)
    return CommandRisk("safe", ())
