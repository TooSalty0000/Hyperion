"""Project data models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ProjectStatus(Enum):
    """Status of a project."""
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StatusHistoryEntry:
    """A single status change entry."""
    status: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[int] = None


@dataclass
class Project:
    """
    Project definition with workspace path and settings.

    Projects allow quick access to workspaces by name,
    with optional default commands and settings.
    """
    name: str
    path: str
    settings: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional default."""
        if self.settings is None:
            return default
        return self.settings.get(key, default)


@dataclass
class ProjectState:
    """
    Current state of a project including status and context.

    This is separate from Project to allow frequent updates
    without modifying the core project definition.
    """
    project_name: str
    status: ProjectStatus = ProjectStatus.IDLE
    current_task: Optional[str] = None
    current_session_id: Optional[int] = None
    last_output_summary: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    progress_percent: int = 0
    history: List[StatusHistoryEntry] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_status(self, status: ProjectStatus, message: str = "", session_id: int = None):
        """Update the project status with history tracking."""
        self.status = status
        self.updated_at = datetime.utcnow()

        # Add to history (keep last 50 entries)
        entry = StatusHistoryEntry(
            status=status.value,
            message=message,
            session_id=session_id
        )
        self.history.append(entry)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def add_blocker(self, blocker: str):
        """Add a blocker."""
        if blocker not in self.blockers:
            self.blockers.append(blocker)
            self.status = ProjectStatus.BLOCKED

    def clear_blockers(self):
        """Clear all blockers."""
        self.blockers.clear()
        if self.status == ProjectStatus.BLOCKED:
            self.status = ProjectStatus.IDLE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_name": self.project_name,
            "status": self.status.value,
            "current_task": self.current_task,
            "current_session_id": self.current_session_id,
            "last_output_summary": self.last_output_summary,
            "blockers": self.blockers,
            "progress_percent": self.progress_percent,
            "history": [
                {
                    "status": h.status,
                    "message": h.message,
                    "timestamp": h.timestamp.isoformat(),
                    "session_id": h.session_id,
                }
                for h in self.history
            ],
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectState':
        """Create from dictionary."""
        history = [
            StatusHistoryEntry(
                status=h["status"],
                message=h["message"],
                timestamp=datetime.fromisoformat(h["timestamp"]),
                session_id=h.get("session_id"),
            )
            for h in data.get("history", [])
        ]

        return cls(
            project_name=data["project_name"],
            status=ProjectStatus(data.get("status", "idle")),
            current_task=data.get("current_task"),
            current_session_id=data.get("current_session_id"),
            last_output_summary=data.get("last_output_summary"),
            blockers=data.get("blockers", []),
            progress_percent=data.get("progress_percent", 0),
            history=history,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )
