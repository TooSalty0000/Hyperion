"""Redis-backed session store implementation."""

import json
import logging
from typing import Optional, List
from datetime import datetime

from .interface import SessionStore, SessionData, SessionStatus

logger = logging.getLogger(__name__)

# Lazy import to avoid requiring redis if not used
_redis = None


def _import_redis():
    """Lazy import redis.asyncio."""
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for RedisSessionStore. "
                "Install it with: pip install redis"
            )
    return _redis


class RedisSessionStore(SessionStore):
    """
    Redis-backed session store.

    Key patterns:
    - vega:session:{session_id} - Session data (JSON)
    - vega:channel_index - Hash: channel_id -> session_id
    - vega:project_index - Hash: project_name -> session_id
    - vega:sessions - Set of all session IDs
    - vega:next_session_id - Counter for next ID
    """

    SESSION_KEY = "vega:session:{session_id}"
    CHANNEL_INDEX = "vega:channel_index"
    PROJECT_INDEX = "vega:project_index"
    SESSION_SET = "vega:sessions"
    NEXT_ID_KEY = "vega:next_session_id"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis session store.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_url = redis_url
        self._redis = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis

    def _serialize_session(self, session: SessionData) -> str:
        """Serialize session to JSON."""
        return json.dumps({
            "session_id": session.session_id,
            "channel_id": session.channel_id,
            "pid": session.pid,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "workspace_dir": session.workspace_dir,
            "command": session.command,
            "project_name": session.project_name
        })

    def _deserialize_session(self, data: str) -> SessionData:
        """Deserialize session from JSON."""
        d = json.loads(data)
        return SessionData(
            session_id=d["session_id"],
            channel_id=d["channel_id"],
            pid=d.get("pid"),
            status=SessionStatus(d["status"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            workspace_dir=d.get("workspace_dir"),
            command=d.get("command"),
            project_name=d.get("project_name")
        )

    async def create(
        self,
        channel_id: int,
        workspace_dir: str = None,
        command: str = None,
        project_name: str = None
    ) -> SessionData:
        """Create a new session record."""
        client = await self._get_client()

        # Atomically get and increment next ID
        session_id = await client.incr(self.NEXT_ID_KEY)

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

        # Store session
        key = self.SESSION_KEY.format(session_id=session_id)
        await client.set(key, self._serialize_session(session))

        # Index by channel
        if channel_id:
            await client.hset(self.CHANNEL_INDEX, str(channel_id), str(session_id))

        # Index by project name
        if project_name:
            await client.hset(self.PROJECT_INDEX, project_name, str(session_id))

        # Add to session set
        await client.sadd(self.SESSION_SET, str(session_id))

        logger.debug(f"Created session {session_id} for channel {channel_id}")
        return session

    async def get(self, session_id: int) -> Optional[SessionData]:
        """Get session by ID."""
        client = await self._get_client()
        key = self.SESSION_KEY.format(session_id=session_id)
        data = await client.get(key)
        if data:
            return self._deserialize_session(data)
        return None

    async def get_by_channel(self, channel_id: int) -> Optional[SessionData]:
        """Get session by channel ID."""
        client = await self._get_client()
        session_id = await client.hget(self.CHANNEL_INDEX, str(channel_id))
        if session_id:
            return await self.get(int(session_id))
        return None

    async def get_by_project(self, project_name: str) -> Optional[SessionData]:
        """Get session by project name."""
        client = await self._get_client()
        session_id = await client.hget(self.PROJECT_INDEX, project_name)
        if session_id:
            return await self.get(int(session_id))
        return None

    async def update(self, session_id: int, **kwargs) -> bool:
        """Update session fields."""
        client = await self._get_client()
        key = self.SESSION_KEY.format(session_id=session_id)

        data = await client.get(key)
        if not data:
            return False

        session = self._deserialize_session(data)

        for field, value in kwargs.items():
            if hasattr(session, field):
                if field == "status" and isinstance(value, str):
                    value = SessionStatus(value)
                setattr(session, field, value)

        await client.set(key, self._serialize_session(session))
        logger.debug(f"Updated session {session_id}: {kwargs}")
        return True

    async def delete(self, session_id: int) -> bool:
        """Delete session record."""
        client = await self._get_client()
        key = self.SESSION_KEY.format(session_id=session_id)

        # Get session to find channel and project
        session = await self.get(session_id)
        if not session:
            return False

        # Remove from channel index
        await client.hdel(self.CHANNEL_INDEX, str(session.channel_id))

        # Remove from project index
        if session.project_name:
            await client.hdel(self.PROJECT_INDEX, session.project_name)

        # Remove from session set
        await client.srem(self.SESSION_SET, str(session_id))

        # Delete session data
        await client.delete(key)

        logger.debug(f"Deleted session {session_id}")
        return True

    async def list_all(self) -> List[SessionData]:
        """List all active sessions."""
        client = await self._get_client()
        session_ids = await client.smembers(self.SESSION_SET)

        sessions = []
        for sid in session_ids:
            session = await self.get(int(sid))
            if session:
                sessions.append(session)

        return sessions

    async def get_next_id(self) -> int:
        """Get next available session ID (without incrementing)."""
        client = await self._get_client()
        current = await client.get(self.NEXT_ID_KEY)
        return int(current) + 1 if current else 1

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
