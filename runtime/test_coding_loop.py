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
    # Plain `npm test`: --runInBand is a Jest-only flag and breaks other runners.
    assert ["npm", "test"] in commands
    assert ["npm", "run", "build"] in commands


def test_detect_node_jest_gets_run_in_band(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}', encoding="utf-8")
    commands = CodingLoop(tmp_path).detect_commands()
    assert ["npm", "test", "--", "--runInBand"] in commands


def test_default_npm_test_script_is_skipped(tmp_path: Path):
    # The npm init placeholder script always exits 1; that is not a real validator.
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"echo \\"Error: no test specified\\" && exit 1"}}',
        encoding="utf-8",
    )
    assert CodingLoop(tmp_path).detect_commands() == []
