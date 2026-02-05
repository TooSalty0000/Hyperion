"""File-based persistent session store."""

import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .interface import SessionStore, SessionData, SessionStatus

logger = logging.getLogger(__name__)


class FileSessionStore(SessionStore):
    """
    File-based session store for persistent storage without Redis.

    Stores session data in a JSON file that survives bot restarts.
    """

    def __init__(self, file_path: str = None):
        """
        Initialize file-based session store.

        Args:
            file_path: Path to JSON file. Defaults to ~/.hyperion/sessions.json
        """
        if file_path:
            self._file_path = Path(file_path)
        else:
            self._file_path = Path.home() / ".hyperion" / "sessions.json"

        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[int, SessionData] = {}
        self._channel_index: Dict[int, int] = {}  # channel_id -> session_id
        self._next_id: int = 1
        self._lock = asyncio.Lock()

        # Load existing sessions on init
        self._load_from_file()

    def _load_from_file(self):
        """Load sessions from file."""
        if not self._file_path.exists():
            logger.info(f"Session file does not exist, starting fresh: {self._file_path}")
            return

        try:
            with open(self._file_path, 'r') as f:
                data = json.load(f)

            self._next_id = data.get('next_id', 1)

            for session_dict in data.get('sessions', []):
                session = SessionData(
                    session_id=session_dict['session_id'],
                    channel_id=session_dict['channel_id'],
                    pid=session_dict.get('pid'),
                    status=SessionStatus(session_dict['status']),
                    created_at=datetime.fromisoformat(session_dict['created_at']),
                    workspace_dir=session_dict.get('workspace_dir'),
                    command=session_dict.get('command'),
                    project_name=session_dict.get('project_name'),
                    notes=session_dict.get('notes'),
                    mode=session_dict.get('mode', 'shell'),
                )
                self._sessions[session.session_id] = session
                if session.channel_id:
                    self._channel_index[session.channel_id] = session.session_id

            logger.info(f"Loaded {len(self._sessions)} sessions from {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to load sessions from file: {e}")

    def _save_to_file(self):
        """Save sessions to file."""
        try:
            data = {
                'next_id': self._next_id,
                'sessions': [
                    {
                        'session_id': s.session_id,
                        'channel_id': s.channel_id,
                        'pid': s.pid,
                        'status': s.status.value,
                        'created_at': s.created_at.isoformat(),
                        'workspace_dir': s.workspace_dir,
                        'command': s.command,
                        'project_name': getattr(s, 'project_name', None),
                        'notes': getattr(s, 'notes', None),
                        'mode': getattr(s, 'mode', 'shell'),
                    }
                    for s in self._sessions.values()
                ]
            }

            with open(self._file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._sessions)} sessions to {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to save sessions to file: {e}")

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
            )
            # Store project name as extra attribute
            session.project_name = project_name

            self._sessions[session_id] = session
            if channel_id:
                self._channel_index[channel_id] = session_id

            self._save_to_file()
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
            if getattr(session, 'project_name', None) == project_name:
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

            self._save_to_file()
            return True

    async def delete(self, session_id: int) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if session.channel_id in self._channel_index:
                del self._channel_index[session.channel_id]
            del self._sessions[session_id]

            self._save_to_file()
            return True

    async def list_all(self) -> List[SessionData]:
        return list(self._sessions.values())

    async def get_next_id(self) -> int:
        return self._next_id

    async def get_running_sessions(self) -> List[SessionData]:
        """Get all sessions with RUNNING status."""
        return [
            s for s in self._sessions.values()
            if s.status == SessionStatus.RUNNING
        ]
