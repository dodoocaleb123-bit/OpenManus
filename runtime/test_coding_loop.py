import asyncio
from pathlib import Path

from app.platform.coding_loop import CodingLoop


def test_detect_python_validator(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    commands = CodingLoop(tmp_path).detect_commands()
    assert commands and "pytest" in " ".join(commands[0])


def test_python_project_without_tests_has_no_pytest_validator(tmp_path: Path):
    # pytest exits 5 ("no tests collected") -> previously failed every cycle.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert CodingLoop(tmp_path).detect_commands() == []


def test_detect_node_build_and_test(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest","build":"vite"}}', encoding="utf-8")
    commands = CodingLoop(tmp_path).detect_commands()
    # --runInBand is Jest-only; vitest/mocha reject it.
    assert ["npm", "test"] in commands
    assert ["npm", "run", "build"] in commands


def test_jest_runs_in_band(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}', encoding="utf-8")
    assert CodingLoop(tmp_path).detect_commands() == [["npm", "test", "--", "--runInBand"]]


def test_npm_placeholder_test_script_is_ignored(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"echo \\"Error: no test specified\\" && exit 1"}}', encoding="utf-8"
    )
    assert CodingLoop(tmp_path).detect_commands() == []


def test_missing_toolchain_is_skipped_not_crashing(tmp_path: Path):
    result = asyncio.run(CodingLoop(tmp_path).run_command(["definitely-not-installed-tool", "test"]))
    assert result.ok and result.skipped


def test_validation_runs_real_pytest_in_project_venv(tmp_path: Path):
    """End-to-end: creates .venv, installs pytest, runs the project's tests."""
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n", encoding="utf-8")
    loop = CodingLoop(tmp_path)
    results = asyncio.run(loop.validate())
    assert (tmp_path / ".venv" / "bin" / "python").exists()
    assert not results[-1].ok and "assert" in results[-1].output
    # Fix the bug; second validation reuses the venv (install stamp) and passes.
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    results = asyncio.run(loop.validate())
    assert all(r.ok for r in results), [r.as_dict() for r in results]
    assert not any(r.command.startswith("(setup)") for r in results)
