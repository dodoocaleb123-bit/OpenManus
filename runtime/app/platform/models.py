from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Project(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    workspace: str
    repository: Optional[str] = None
    branch: Optional[str] = None
    git_ready: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    project_id: str
    prompt: str
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 0
    checkpoint: Optional[str] = None
    browser_session_id: Optional[str] = None
    coding_iteration: int = 0
    validation: Optional[dict[str, Any]] = None


class BrowserSessionInfo(BaseModel):
    id: str
    project_id: str
    running: bool = False
    url: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class Event(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_id: str
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
