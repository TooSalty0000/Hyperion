"""Data models for the Polaris todo system."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class TodoStatus(str, Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class TodoPriority(int, Enum):
    """Priority level of a todo item."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TodoItem:
    """A single todo item with full metadata."""

    id: str
    user_id: str
    title: str
    status: TodoStatus = TodoStatus.PENDING
    priority: TodoPriority = TodoPriority.MEDIUM
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    project_id: Optional[str] = None
    parent_id: Optional[str] = None  # for subtasks
    due_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    estimated_duration_minutes: Optional[int] = None
    calendar_event_id: Optional[str] = None
    reminder_ids: List[str] = field(default_factory=list)
    reschedule_count: int = 0
    version: int = 1  # optimistic concurrency
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority.value,
            "description": self.description,
            "tags": self.tags,
            "project_id": self.project_id,
            "parent_id": self.parent_id,
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "calendar_event_id": self.calendar_event_id,
            "reminder_ids": self.reminder_ids,
            "reschedule_count": self.reschedule_count,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            title=data["title"],
            status=TodoStatus(data.get("status", "pending")),
            priority=TodoPriority(data.get("priority", 2)),
            description=data.get("description"),
            tags=data.get("tags", []),
            project_id=data.get("project_id"),
            parent_id=data.get("parent_id"),
            due_at=datetime.fromisoformat(data["due_at"]) if data.get("due_at") else None,
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            estimated_duration_minutes=data.get("estimated_duration_minutes"),
            calendar_event_id=data.get("calendar_event_id"),
            reminder_ids=data.get("reminder_ids", []),
            reschedule_count=data.get("reschedule_count", 0),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {}),
        )

    def format_for_display(self) -> str:
        """Format todo for Discord/LLM display."""
        status_icons = {
            TodoStatus.PENDING: "[ ]",
            TodoStatus.IN_PROGRESS: "[~]",
            TodoStatus.COMPLETED: "[x]",
            TodoStatus.CANCELLED: "[-]",
            TodoStatus.DEFERRED: "[>]",
        }
        priority_icons = {
            TodoPriority.LOW: "",
            TodoPriority.MEDIUM: "!",
            TodoPriority.HIGH: "!!",
            TodoPriority.CRITICAL: "!!!",
        }

        icon = status_icons.get(self.status, "[ ]")
        pri = priority_icons.get(self.priority, "")
        pri_str = f" ({pri})" if pri else ""

        parts = [f"{icon} **{self.title}**{pri_str}"]

        if self.due_at:
            parts.append(f"  Due: {self.due_at.strftime('%b %d, %Y %I:%M %p')}")
        if self.scheduled_at:
            parts.append(f"  Scheduled: {self.scheduled_at.strftime('%b %d, %Y %I:%M %p')}")
        if self.tags:
            parts.append(f"  Tags: {', '.join(self.tags)}")
        if self.description:
            desc = self.description[:80] + "..." if len(self.description) > 80 else self.description
            parts.append(f"  {desc}")

        parts.append(f"  [id: {self.id}]")
        return "\n".join(parts)

    @property
    def is_overdue(self) -> bool:
        """Check if this todo is past due."""
        if not self.due_at:
            return False
        if self.status in (TodoStatus.COMPLETED, TodoStatus.CANCELLED):
            return False
        return datetime.utcnow() > self.due_at


@dataclass
class TodoProject:
    """A project that groups todo items."""

    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = field(default_factory=lambda: datetime.utcnow())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoProject":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            description=data.get("description"),
            color=data.get("color"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )
