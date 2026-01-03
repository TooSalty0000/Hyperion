from typing import Optional, List, Dict
from datetime import datetime
import asyncio

from .interface import SessionStore, SessionData, SessionStatus


class InMemorySessionStore(SessionStore):
    """In-memory session store for development/single-instance deployment."""

    def __init__(self):
        self._sessions: Dict[int, SessionData] = {}
        self._channel_index: Dict[int, int] = {}  # channel_id -> session_id
        self._next_id: int = 1
        self._lock = asyncio.Lock()

    async def create(self, channel_id: int, workspace_dir: str = None,
                     command: str = None, project_name: str = None) -> SessionData:
        async with self._lock:
            session_id = self._next_id
            self._next_id += 1

            session = SessionData(
                session_id=session_id,
                channel_id=channel_id,
                pid=None,
                status=SessionStatus.STARTING,
                created_at=datetime.utcnow(),
                workspace_dir=workspace_dir,
                command=command,
                project_name=project_name
            )

            self._sessions[session_id] = session
            if channel_id:
                self._channel_index[channel_id] = session_id
            return session

    async def get(self, session_id: int) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    async def get_by_channel(self, channel_id: int) -> Optional[SessionData]:
        session_id = self._channel_index.get(channel_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def get_by_project(self, project_name: str) -> Optional[SessionData]:
        """Get session by project name."""
        for session in self._sessions.values():
            if session.project_name == project_name:
                return session
        return None

    async def update(self, session_id: int, **kwargs) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            # Update allowed fields
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            return True

    async def delete(self, session_id: int) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if session.channel_id in self._channel_index:
                del self._channel_index[session.channel_id]
            del self._sessions[session_id]
            return True

    async def list_all(self) -> List[SessionData]:
        return list(self._sessions.values())

    async def get_next_id(self) -> int:
        return self._next_id
