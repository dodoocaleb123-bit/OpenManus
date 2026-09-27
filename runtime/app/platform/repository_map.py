from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

IGNORED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build"}
IGNORED_FILES = {".env", ".env.local", ".env.production", "config.toml", "config.json"}
MAX_FILES = 1200
MAX_BYTES_PER_FILE = 200_000


def _language(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".go": "go", ".rs": "rust", ".java": "java",
        ".md": "markdown", ".json": "json", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    }.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "other")


def _python_symbols(path: Path) -> dict[str, list[str]]:
    try:
        if path.stat().st_size > MAX_BYTES_PER_FILE:
            return {}
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {}
    return {
        "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
        "functions": [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
    }


def build_repository_map(root: str | Path) -> dict[str, Any]:
    """Return a bounded, safe structural map; never reads file contents."""
    workspace = Path(root).resolve()
    files: list[dict[str, Any]] = []
    languages: Counter[str] = Counter()
    for path in sorted(workspace.rglob("*")):
        if len(files) >= MAX_FILES:
            break
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in path.relative_to(workspace).parts):
            continue
        if path.name in IGNORED_FILES or path.name.endswith(".sqlite") or path.name.endswith(".db"):
            continue
        relative = str(path.relative_to(workspace))
        language = _language(path)
        languages[language] += 1
        record: dict[str, Any] = {"path": relative, "language": language, "bytes": path.stat().st_size}
        if path.suffix.lower() == ".py":
            symbols = _python_symbols(path)
            if symbols:
                record["symbols"] = symbols
        files.append(record)
    return {
        "workspace": str(workspace),
        "file_count": len(files),
        "truncated": len(files) >= MAX_FILES,
        "languages": dict(sorted(languages.items())),
        "files": files,
    }
