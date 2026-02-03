"""Pydantic request/response schemas for the Polaris API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# === Auth ===

class LoginRequest(BaseModel):
    password: str


class TokenResponse(BaseModel):
    token: str
    expires_in_days: int = 30


class AuthStatus(BaseModel):
    authenticated: bool
    user_id: str


# === Todos ===

class TodoStatusEnum(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"
    deferred = "deferred"


class TodoPriorityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


PRIORITY_MAP = {
    "low": 1, "medium": 2, "high": 3, "critical": 4,
}
PRIORITY_REVERSE = {1: "low", 2: "medium", 3: "high", 4: "critical"}


class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: TodoPriorityEnum = TodoPriorityEnum.medium
    tags: List[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    parent_id: Optional[str] = None
    due_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    estimated_duration_minutes: Optional[int] = None


class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TodoStatusEnum] = None
    priority: Optional[TodoPriorityEnum] = None
    tags: Optional[List[str]] = None
    project_id: Optional[str] = None
    due_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    estimated_duration_minutes: Optional[int] = None
    clear_due: bool = False
    clear_scheduled: bool = False


class TodoResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str] = None
    status: str
    priority: int
    priority_label: str
    tags: List[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    parent_id: Optional[str] = None
    due_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    estimated_duration_minutes: Optional[int] = None
    calendar_event_id: Optional[str] = None
    reschedule_count: int = 0
    version: int = 1
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    is_overdue: bool = False

    @classmethod
    def from_todo(cls, todo) -> "TodoResponse":
        return cls(
            id=todo.id,
            user_id=todo.user_id,
            title=todo.title,
            description=todo.description,
            status=todo.status.value,
            priority=todo.priority.value,
            priority_label=PRIORITY_REVERSE.get(todo.priority.value, "medium"),
            tags=todo.tags,
            project_id=todo.project_id,
            parent_id=todo.parent_id,
            due_at=todo.due_at,
            scheduled_at=todo.scheduled_at,
            estimated_duration_minutes=todo.estimated_duration_minutes,
            calendar_event_id=todo.calendar_event_id,
            reschedule_count=todo.reschedule_count,
            version=todo.version,
            created_at=todo.created_at,
            updated_at=todo.updated_at,
            completed_at=todo.completed_at,
            is_overdue=todo.is_overdue,
        )


class TodoSummaryResponse(BaseModel):
    total: int
    by_status: Dict[str, int]
    overdue_count: int
    due_today_count: int


# === Reminders ===

class ReminderCreate(BaseModel):
    message: str
    fire_at: datetime
    todo_id: Optional[str] = None


class ReminderResponse(BaseModel):
    id: str
    user_id: str
    todo_id: Optional[str] = None
    message: str
    fire_at: datetime
    status: str
    snooze_count: int = 0
    created_at: datetime
    fired_at: Optional[datetime] = None

    @classmethod
    def from_reminder(cls, r) -> "ReminderResponse":
        return cls(
            id=r.id,
            user_id=r.user_id,
            todo_id=r.todo_id,
            message=r.message,
            fire_at=r.fire_at,
            status=r.status.value,
            snooze_count=r.snooze_count,
            created_at=r.created_at,
            fired_at=r.fired_at,
        )


# === Push ===

class PushSubscription(BaseModel):
    endpoint: str
    keys: Dict[str, str]  # p256dh, auth


class VapidKeyResponse(BaseModel):
    public_key: str
