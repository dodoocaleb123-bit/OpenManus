from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class PlatformObservability:
    """Small, dependency-free operational summary for a single-node deployment.

    The platform already persists task/event state in SQLite and redacted audit
    records in JSONL. This class deliberately derives metrics from those durable
    sources instead of introducing another database or an in-memory-only counter.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _audit_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        path = self.root / "audit.jsonl"
        if not path.is_file():
            return {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    action = json.loads(line).get("action")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if action:
                    counts[str(action)] += 1
        except OSError:
            return {}
        return dict(sorted(counts.items()))

    def snapshot(self, store: Any) -> dict[str, Any]:
        tasks = list(store.tasks.values())
        statuses = Counter(task.status.value for task in tasks)
        recovery = Counter((task.recovery or {}).get("category", "unknown") for task in tasks if task.recovery)
        validation = [task.evidence.get("verification", {}).get("passed") for task in tasks if task.evidence]
        validated = [value for value in validation if isinstance(value, bool)]
        projects = list(store.projects.values())
        return {
            "projects": len(projects),
            "tasks": len(tasks),
            "task_statuses": dict(sorted(statuses.items())),
            "recovery_categories": dict(sorted(recovery.items())),
            "verification": {
                "recorded": len(validated),
                "passed": sum(value is True for value in validated),
                "failed": sum(value is False for value in validated),
            },
            "latency": store.latency_metrics(),
            "feedback": store.feedback_metrics(),
            "audit_actions": self._audit_counts(),
            "limits": {
                "single_node": True,
                "persistent_store": "sqlite",
                "task_isolation": "process/container isolation is not enabled by this release",
            },
        }
