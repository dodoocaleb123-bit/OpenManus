import asyncio
from pathlib import Path

from app.platform.coding_loop import CodingLoop


def test_detect_python_validator(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    commands = CodingLoop(tmp_path).detect_commands()
    assert commands and commands[0][0] == "pytest"


def test_detect_node_build_and_test(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest","build":"vite"}}', encoding="utf-8")
    commands = CodingLoop(tmp_path).detect_commands()
    assert ["npm", "test", "--", "--runInBand"] in commands
    assert ["npm", "run", "build"] in commands
