"""Data models for the Polaris reminder system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any


class ReminderType(str, Enum):
    """How to deliver the reminder."""
    DISCORD_DM = "discord_dm"
    WEB_PUSH = "web_push"
    CHANNEL_MENTION = "channel_mention"


class ReminderStatus(str, Enum):
    """Reminder lifecycle status."""
    PENDING = "pending"
    FIRED = "fired"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"


@dataclass
class ReminderItem:
    """A scheduled reminder tied to a todo or standalone."""

    id: str
    user_id: str
    todo_id: Optional[str]  # None for standalone reminders
    message: str  # What to remind about
    fire_at: datetime  # When to fire
    status: ReminderStatus = ReminderStatus.PENDING
    reminder_types: List[ReminderType] = field(
        default_factory=lambda: [ReminderType.CHANNEL_MENTION, ReminderType.WEB_PUSH]
    )
    snooze_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fired_at: Optional[datetime] = None
    channel_id: Optional[int] = None  # For channel_mention type
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "todo_id": self.todo_id,
            "message": self.message,
            "fire_at": self.fire_at.isoformat(),
            "status": self.status.value,
            "reminder_types": [rt.value for rt in self.reminder_types],
            "snooze_count": self.snooze_count,
            "created_at": self.created_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "channel_id": self.channel_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReminderItem":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            todo_id=data.get("todo_id"),
            message=data["message"],
            fire_at=datetime.fromisoformat(data["fire_at"]),
            status=ReminderStatus(data.get("status", "pending")),
            reminder_types=[ReminderType(rt) for rt in data.get("reminder_types", ["discord_dm"])],
            snooze_count=data.get("snooze_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            fired_at=datetime.fromisoformat(data["fired_at"]) if data.get("fired_at") else None,
            channel_id=data.get("channel_id"),
            metadata=data.get("metadata", {}),
        )
