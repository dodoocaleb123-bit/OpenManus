"""Small, dependency-free workspace retrieval for local project context.

This is a bounded lexical selector rather than an embedding service: it keeps
Ollama-only deployments offline-capable and returns file paths with excerpts so
answers can be grounded in inspectable workspace evidence.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", "coverage", ".next", "target", "vendor",
}
SECRET_NAMES = {".env", ".env.local", ".env.production", "config.toml", "config.json"}
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html",
    ".htm", ".css", ".scss", ".vue", ".svelte", ".go", ".rs", ".java",
    ".kt", ".swift", ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb",
    ".sh", ".sql", ".md", ".rst", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".xml", ".dockerfile",
}
PREFERRED_NAMES = {
    "readme.md", "readme", "package.json", "pyproject.toml", "requirements.txt",
    "cargo.toml", "go.mod", "dockerfile",
}
TOKEN_RE = re.compile(r"[a-zA-Z0-9_./-]{3,}")
SECRET_FILE_RE = re.compile(r"(?i)(secret|credential|password|private[-_]?key|id_rsa|\.pem$|\.key$)")
MAX_FILES_SCANNED = 1000
MAX_FILE_BYTES = 100_000
MAX_SCAN_BYTES = 3_000_000
PRIOR_CONTEXT_RE = re.compile(
    r"\b(?:continue|previous|earlier|before|above|prior|same|again|as\s+we\s+discussed|"
    r"what\s+i\s+(?:said|sent)|that\s+(?:project|page|file|image|design|answer|task))\b",
    re.IGNORECASE,
)


def _query_tokens(query: str) -> set[str]:
    return {token.casefold().strip("./-") for token in TOKEN_RE.findall(query) if len(token.strip("./-")) >= 3}


def _iter_files(root: Path) -> Iterable[Path]:
    files: list[Path] = []
    try:
        for directory, subdirectories, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(directory)
            subdirectories[:] = sorted(
                name for name in subdirectories
                if name not in IGNORED_DIRECTORIES and not (current / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = current / filename
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    name = path.name.casefold()
                    if name in SECRET_NAMES or (name.startswith(".env.") and name not in {".env.example", ".env.sample"}):
                        continue
                    if SECRET_FILE_RE.search(name):
                        continue
                    if path.suffix.casefold() not in TEXT_EXTENSIONS and name not in PREFERRED_NAMES and name != "dockerfile":
                        continue
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                    files.append(path)
                    if len(files) >= MAX_FILES_SCANNED:
                        return files
                except OSError:
                    continue
    except OSError:
        return files
    return files


def build_workspace_context(
    root: str | Path,
    query: str = "",
    *,
    max_files: int = 5,
    max_chars: int = 18_000,
) -> str:
    """Return a bounded file tree and query-relevant excerpts from ``root``.

    Secret-like filenames, symlinks, generated directories, binary/large files,
    and unsupported file types are excluded. File contents are explicitly
    identified as untrusted project data by callers before being sent to a model.
    """
    workspace = Path(root).resolve()
    files = list(_iter_files(workspace))
    if not files:
        return "Repository file tree (bounded):\n(workspace has no readable source files)"

    max_chars = max(1200, int(max_chars))
    tree_budget = min(6000, max_chars // 3)
    tree = "\n".join(str(path.relative_to(workspace)) for path in files[:300])
    if len(tree) > tree_budget:
        tree = tree[:tree_budget].rsplit("\n", 1)[0] + "\n… (tree truncated)"
    sections = [f"Repository file tree (bounded):\n{tree}"]
    terms = _query_tokens(query)
    preferred = [path for path in files if path.name.casefold() in PREFERRED_NAMES or path.name.casefold() == "dockerfile"]

    scored: list[tuple[int, str, Path, str]] = []
    scanned = 0
    for path in files:
        try:
            size = path.stat().st_size
            if scanned + size > MAX_SCAN_BYTES:
                continue
            raw = path.read_bytes()
            scanned += len(raw)
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(workspace))
        lower_path = rel.casefold()
        lower_text = text.casefold()
        score = sum(5 for token in terms if token in lower_path)
        score += sum(min(12, lower_text.count(token)) for token in terms)
        if path in preferred:
            score += 2
        if not terms and path in preferred:
            score += 10
        if score:
            scored.append((score, rel, path, text))

    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    chosen = scored[: max(1, min(int(max_files), 8))]
    # If a repository is tiny or the query does not match source text, still
    # provide its small set of high-signal manifests/readmes for useful context.
    if not chosen:
        chosen = [(0, str(path.relative_to(workspace)), path, path.read_text(encoding="utf-8", errors="replace")) for path in preferred[:max_files]]

    remaining = max(0, int(max_chars) - len("\n".join(sections)))
    if chosen and remaining:
        sections.append("\nProject file excerpts (ranked by lexical relevance; paths are provenance):")
    for _, rel, _, text in chosen:
        if remaining <= 0:
            break
        excerpt = text[: min(6_000, remaining)]
        if not excerpt:
            continue
        section = f"\n--- {rel} ---\n{excerpt}"
        sections.append(section)
        remaining -= len(section)
    return "\n".join(sections)


def task_conversation_context(messages: Iterable[object], current_prompt: str, task_id: str) -> str:
    """Return current-task history unless the user explicitly requests continuity."""
    items = list(messages)
    current = [item for item in items if getattr(item, "task_id", None) == task_id]
    if not PRIOR_CONTEXT_RE.search(current_prompt):
        selected = current
    else:
        prior = [item for item in items if getattr(item, "task_id", None) != task_id]
        selected = prior[-10:] + current
    return "\n".join(
        f"{getattr(item, 'role', 'unknown').upper()}: {getattr(item, 'content', '')}"
        for item in selected
        if getattr(item, "content", None)
    )[-30000:] or "(no prior task conversation)"
