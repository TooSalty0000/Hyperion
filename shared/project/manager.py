"""Redis-backed project management."""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import Project

logger = logging.getLogger(__name__)

# Lazy import
_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for ProjectManager. "
                "Install it with: pip install redis"
            )
    return _redis


class ProjectManager:
    """
    Redis-backed project management.

    Key patterns:
    - vega:project:{name} - Project data (JSON)
    - vega:projects - Set of project names
    """

    PROJECT_KEY = "vega:project:{name}"
    PROJECT_SET = "vega:projects"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize project manager.

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

    def _serialize(self, project: Project) -> str:
        """Serialize project to JSON."""
        return json.dumps({
            "name": project.name,
            "path": project.path,
            "settings": project.settings,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat()
        })

    def _deserialize(self, data: str) -> Project:
        """Deserialize project from JSON."""
        d = json.loads(data)
        return Project(
            name=d["name"],
            path=d["path"],
            settings=d.get("settings"),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"])
        )

    async def create(
        self,
        name: str,
        path: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Project:
        """
        Create a new project.

        Args:
            name: Unique project name
            path: Filesystem path to project
            settings: Optional settings dict

        Returns:
            Created Project

        Raises:
            ValueError: If project with name already exists
        """
        client = await self._get_client()

        # Check if exists
        existing = await self.get(name)
        if existing:
            raise ValueError(f"Project '{name}' already exists")

        project = Project(
            name=name,
            path=path,
            settings=settings
        )

        key = self.PROJECT_KEY.format(name=name)
        await client.set(key, self._serialize(project))
        await client.sadd(self.PROJECT_SET, name)

        logger.info(f"Created project '{name}' at {path}")
        return project

    async def get(self, name: str) -> Optional[Project]:
        """Get project by name."""
        client = await self._get_client()
        key = self.PROJECT_KEY.format(name=name)
        data = await client.get(key)
        if data:
            return self._deserialize(data)
        return None

    async def update(self, name: str, **kwargs) -> Optional[Project]:
        """
        Update project fields.

        Args:
            name: Project name
            **kwargs: Fields to update (path, settings)

        Returns:
            Updated Project or None if not found
        """
        client = await self._get_client()
        project = await self.get(name)
        if not project:
            return None

        for field, value in kwargs.items():
            if hasattr(project, field):
                setattr(project, field, value)

        project.updated_at = datetime.utcnow()

        key = self.PROJECT_KEY.format(name=name)
        await client.set(key, self._serialize(project))

        logger.info(f"Updated project '{name}'")
        return project

    async def delete(self, name: str) -> bool:
        """Delete a project."""
        client = await self._get_client()
        key = self.PROJECT_KEY.format(name=name)

        result = await client.delete(key)
        await client.srem(self.PROJECT_SET, name)

        if result > 0:
            logger.info(f"Deleted project '{name}'")
            return True
        return False

    async def list_all(self) -> List[Project]:
        """List all projects."""
        client = await self._get_client()
        names = await client.smembers(self.PROJECT_SET)

        projects = []
        for name in sorted(names):
            project = await self.get(name)
            if project:
                projects.append(project)

        return projects

    async def exists(self, name: str) -> bool:
        """Check if project exists."""
        client = await self._get_client()
        return await client.sismember(self.PROJECT_SET, name)

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
