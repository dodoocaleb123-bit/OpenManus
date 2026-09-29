from app.platform.context import build_workspace_context


def test_context_ranks_relevant_source_and_bounds_output(tmp_path):
    (tmp_path / "README.md").write_text("A small sample project.", encoding="utf-8")
    (tmp_path / "auth.py").write_text("def verify_password(value):\n    return value == 'example'\n", encoding="utf-8")
    (tmp_path / "billing.py").write_text("def invoice_total(rows):\n    return sum(rows)\n", encoding="utf-8")

    context = build_workspace_context(tmp_path, "Where is password verification implemented?", max_files=2, max_chars=1200)

    assert "auth.py" in context
    assert "verify_password" in context
    assert len(context) <= 1400


def test_context_excludes_secrets_symlinks_and_generated_directories(tmp_path):
    (tmp_path / "README.md").write_text("Safe overview.", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=not-for-context", encoding="utf-8")
    (tmp_path / "credentials.json").write_text('{"password":"not-for-context"}', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "safe.py").write_text("def hello(): return 'world'", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "generated.js").write_text("generated", encoding="utf-8")

    context = build_workspace_context(tmp_path, "hello credentials", max_files=5)

    assert "safe.py" in context
    assert "not-for-context" not in context
    assert "generated.js" not in context
