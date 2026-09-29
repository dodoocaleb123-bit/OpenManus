from __future__ import annotations

from pathlib import Path

from app.platform.artifacts import artifact_record, artifact_summary, discover_artifacts, workspace_baseline
from app.platform.sources import citation_records


def test_artifact_discovery_records_changed_files(tmp_path: Path):
    baseline = workspace_baseline(tmp_path)
    output = tmp_path / "report.md"
    output.write_text("# Findings\n", encoding="utf-8")
    artifacts = discover_artifacts(tmp_path, baseline=baseline, task_id="task-1")
    assert len(artifacts) == 1
    assert artifacts[0]["path"] == "report.md"
    assert artifacts[0]["kind"] == "document"
    assert artifacts[0]["task_id"] == "task-1"
    assert artifact_summary(artifacts)["count"] == 1


def test_citations_are_deduplicated_and_timestamped():
    citations = citation_records({"url": "https://example.com/page", "text": "Example page"})
    assert len(citations) == 1
    assert citations[0]["url"] == "https://example.com/page"
    assert citations[0]["retrieved_at"]
    assert citations[0]["confidence"] == "source_recorded"


def test_artifact_record_has_checksum(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nA,1\n", encoding="utf-8")
    record = artifact_record(path, tmp_path)
    assert record["kind"] == "spreadsheet"
    assert len(record["sha256"]) == 64


def test_gui_has_no_request_mode_selector():
    html = Path(__file__).parent.joinpath("web/index.html").read_text(encoding="utf-8")
    assert 'id="executionMode"' not in html
    assert "Request mode" not in html
    assert "mode:selectedMode" not in html


def test_gui_has_compact_chat_loading_indicator():
    html = Path(__file__).parent.joinpath("web/index.html").read_text(encoding="utf-8")
    assert "chat-loading-spinner" in html
    assert "Open Manus" in html
    assert "showChatLoading()" in html
    assert "clearChatReplyState()" in html
