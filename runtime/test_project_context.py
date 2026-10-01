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


def test_task_conversation_context_isolates_unrelated_tasks():
    from types import SimpleNamespace
    from app.platform.context import task_conversation_context

    messages = [
        SimpleNamespace(task_id="math-task", role="user", content="Solve the algebra problem."),
        SimpleNamespace(task_id="math-task", role="assistant", content="The answer is 42."),
        SimpleNamespace(task_id="beauty-task", role="user", content="Create a beauty cosmetics webpage."),
    ]

    context = task_conversation_context(messages, "Create a beauty cosmetics webpage.", "beauty-task")

    assert "beauty cosmetics" in context
    assert "algebra" not in context
    assert "42" not in context


def test_task_conversation_context_allows_explicit_continuity():
    from types import SimpleNamespace
    from app.platform.context import task_conversation_context

    messages = [
        SimpleNamespace(task_id="old-task", role="user", content="Use the pink cosmetics palette."),
        SimpleNamespace(task_id="old-task", role="assistant", content="The cosmetics design uses blush and rose tones."),
        SimpleNamespace(task_id="unrelated-task", role="user", content="Solve the quadratic equation."),
        SimpleNamespace(task_id="new-task", role="user", content="Continue the previous design."),
    ]

    context = task_conversation_context(messages, "Continue the previous design.", "new-task")

    assert "pink cosmetics palette" in context
    assert "Continue the previous design" in context
    assert "quadratic equation" not in context


def test_task_conversation_context_does_not_import_unmatched_explicit_history():
    from types import SimpleNamespace
    from app.platform.context import task_conversation_context

    messages = [
        SimpleNamespace(task_id="old-task", role="user", content="Continue the database migration."),
        SimpleNamespace(task_id="new-task", role="user", content="Continue the previous design."),
    ]

    context = task_conversation_context(messages, "Continue the previous design.", "new-task")

    assert "database migration" not in context
