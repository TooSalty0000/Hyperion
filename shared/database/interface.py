from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SessionStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SessionData:
    """Persistent session metadata."""
    session_id: int          # Managed sequential ID (1, 2, 3, ...)
    channel_id: int          # Discord channel ID
    pid: Optional[int]       # OS process ID (None if not started)
    status: SessionStatus
    created_at: datetime
    workspace_dir: Optional[str] = None
    command: Optional[str] = None
    project_name: Optional[str] = None  # Associated project name for session lookup


class SessionStore(ABC):
    """Abstract interface for session persistence."""

    @abstractmethod
    async def create(self, channel_id: int, workspace_dir: str = None,
                     command: str = None) -> SessionData:
        """Create a new session record, returns assigned session_id."""
        pass

    @abstractmethod
    async def get(self, session_id: int) -> Optional[SessionData]:
        """Get session by managed ID."""
        pass

    @abstractmethod
    async def get_by_channel(self, channel_id: int) -> Optional[SessionData]:
        """Get session by Discord channel ID."""
        pass

    @abstractmethod
    async def get_by_project(self, project_name: str) -> Optional[SessionData]:
        """Get session by project name."""
        pass

    @abstractmethod
    async def update(self, session_id: int, **kwargs) -> bool:
        """Update session fields. Returns success."""
        pass

    @abstractmethod
    async def delete(self, session_id: int) -> bool:
        """Delete session record. Returns success."""
        pass

    @abstractmethod
    async def list_all(self) -> List[SessionData]:
        """List all active sessions."""
        pass

    @abstractmethod
    async def get_next_id(self) -> int:
        """Get next available session ID."""
        pass
