"""Redis-backed todo management for Polaris."""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from polaris.todo.models import TodoItem, TodoProject, TodoStatus, TodoPriority

logger = logging.getLogger(__name__)

_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for TodoManager. "
                "Install it with: pip install redis"
            )
    return _redis


class TodoManager:
    """
    Redis-backed todo CRUD manager.

    Key patterns:
    - hyperion:todo:item:{todo_id}                   -> JSON TodoItem
    - hyperion:todo:user:{user_id}:all               -> SET of todo_ids
    - hyperion:todo:user:{user_id}:status:{status}   -> SET of todo_ids
    - hyperion:todo:user:{user_id}:project:{project}  -> SET of todo_ids
    - hyperion:todo:user:{user_id}:tag:{tag}          -> SET of todo_ids
    - hyperion:todo:user:{user_id}:by_due             -> ZSET: todo_id -> due_timestamp
    - hyperion:todo:user:{user_id}:by_priority        -> ZSET: todo_id -> priority_score
    - hyperion:todo:user:{user_id}:by_created         -> ZSET: todo_id -> created_timestamp
    - hyperion:todo:item:{todo_id}:subtasks           -> LIST of child todo_ids
    - hyperion:todo:counter                           -> INT (INCR for IDs)
    - hyperion:todo:events:{user_id}                  -> Pub/Sub channel for real-time updates
    """

    # Key patterns
    ITEM_KEY = "hyperion:todo:item:{todo_id}"
    ALL_SET = "hyperion:todo:user:{user_id}:all"
    STATUS_SET = "hyperion:todo:user:{user_id}:status:{status}"
    PROJECT_SET = "hyperion:todo:user:{user_id}:project:{project}"
    TAG_SET = "hyperion:todo:user:{user_id}:tag:{tag}"
    BY_DUE = "hyperion:todo:user:{user_id}:by_due"
    BY_PRIORITY = "hyperion:todo:user:{user_id}:by_priority"
    BY_CREATED = "hyperion:todo:user:{user_id}:by_created"
    SUBTASKS_KEY = "hyperion:todo:item:{todo_id}:subtasks"
    COUNTER_KEY = "hyperion:todo:counter"
    EVENTS_CHANNEL = "hyperion:todo:events:{user_id}"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None
        self._completion_tracker = None  # Set by cog if adaptation engine is available

    async def _get_client(self):
        """Get or create Redis client."""
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def _next_id(self) -> str:
        """Generate next todo ID."""
        redis = await self._get_client()
        counter = await redis.incr(self.COUNTER_KEY)
        return f"todo-{counter}"

    async def _publish_event(self, user_id: str, event_type: str, todo_id: str, data: Dict[str, Any] = None):
        """Publish a change event to Redis Pub/Sub for real-time updates."""
        redis = await self._get_client()
        channel = self.EVENTS_CHANNEL.format(user_id=user_id)
        event = {
            "type": event_type,
            "todo_id": todo_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {},
        }
        try:
            await redis.publish(channel, json.dumps(event))
        except Exception as e:
            logger.debug(f"Failed to publish event (non-critical): {e}")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        user_id: str,
        title: str,
        description: Optional[str] = None,
        priority: TodoPriority = TodoPriority.MEDIUM,
        tags: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        due_at: Optional[datetime] = None,
        scheduled_at: Optional[datetime] = None,
        estimated_duration_minutes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TodoItem:
        """Create a new todo item."""
        redis = await self._get_client()
        todo_id = await self._next_id()

        todo = TodoItem(
            id=todo_id,
            user_id=user_id,
            title=title,
            description=description,
            priority=priority,
            tags=tags or [],
            project_id=project_id,
            parent_id=parent_id,
            due_at=due_at,
            scheduled_at=scheduled_at,
            estimated_duration_minutes=estimated_duration_minutes,
            metadata=metadata or {},
        )

        # Store the item
        item_key = self.ITEM_KEY.format(todo_id=todo_id)
        await redis.set(item_key, json.dumps(todo.to_dict()))

        # Add to index sets
        all_key = self.ALL_SET.format(user_id=user_id)
        await redis.sadd(all_key, todo_id)

        status_key = self.STATUS_SET.format(user_id=user_id, status=todo.status.value)
        await redis.sadd(status_key, todo_id)

        # Priority sorted set (higher = more important)
        priority_key = self.BY_PRIORITY.format(user_id=user_id)
        await redis.zadd(priority_key, {todo_id: todo.priority.value})

        # Created sorted set
        created_key = self.BY_CREATED.format(user_id=user_id)
        await redis.zadd(created_key, {todo_id: todo.created_at.timestamp()})

        # Due date sorted set
        if due_at:
            due_key = self.BY_DUE.format(user_id=user_id)
            await redis.zadd(due_key, {todo_id: due_at.timestamp()})

        # Project index
        if project_id:
            proj_key = self.PROJECT_SET.format(user_id=user_id, project=project_id)
            await redis.sadd(proj_key, todo_id)

        # Tag indexes
        for tag in (tags or []):
            tag_key = self.TAG_SET.format(user_id=user_id, tag=tag.lower())
            await redis.sadd(tag_key, todo_id)

        # Subtask relationship
        if parent_id:
            subtask_key = self.SUBTASKS_KEY.format(todo_id=parent_id)
            await redis.rpush(subtask_key, todo_id)

        await self._publish_event(user_id, "created", todo_id, todo.to_dict())

        logger.info(f"Created todo {todo_id} for user {user_id}: {title[:50]}")
        return todo

    async def get(self, todo_id: str) -> Optional[TodoItem]:
        """Get a single todo by ID."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(todo_id=todo_id)
        data = await redis.get(item_key)
        if not data:
            return None
        return TodoItem.from_dict(json.loads(data))

    async def get_batch(self, todo_ids: List[str]) -> List[TodoItem]:
        """Fetch multiple todos in a single mget call."""
        if not todo_ids:
            return []
        redis = await self._get_client()
        keys = [self.ITEM_KEY.format(todo_id=tid) for tid in todo_ids]
        results = await redis.mget(keys)
        return [TodoItem.from_dict(json.loads(d)) for d in results if d]

    async def update(
        self,
        todo_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[TodoStatus] = None,
        priority: Optional[TodoPriority] = None,
        tags: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        due_at: Optional[datetime] = None,
        scheduled_at: Optional[datetime] = None,
        estimated_duration_minutes: Optional[int] = None,
        calendar_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clear_due: bool = False,
        clear_scheduled: bool = False,
    ) -> Optional[TodoItem]:
        """
        Update a todo item. Returns updated item or None if not found.

        Uses optimistic concurrency via version field.
        """
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(todo_id=todo_id)
        data = await redis.get(item_key)
        if not data:
            return None

        todo = TodoItem.from_dict(json.loads(data))
        old_status = todo.status
        old_priority = todo.priority
        old_tags = set(todo.tags)
        old_project = todo.project_id

        # Apply updates
        if title is not None:
            todo.title = title
        if description is not None:
            todo.description = description
        if status is not None:
            todo.status = status
            if status == TodoStatus.COMPLETED:
                todo.completed_at = datetime.utcnow()
        if priority is not None:
            todo.priority = priority
        if tags is not None:
            todo.tags = tags
        if project_id is not None:
            todo.project_id = project_id
        if due_at is not None:
            todo.due_at = due_at
        if clear_due:
            todo.due_at = None
        if scheduled_at is not None:
            todo.scheduled_at = scheduled_at
        if clear_scheduled:
            todo.scheduled_at = None
        if estimated_duration_minutes is not None:
            todo.estimated_duration_minutes = estimated_duration_minutes
        if calendar_event_id is not None:
            todo.calendar_event_id = calendar_event_id
        if metadata is not None:
            todo.metadata.update(metadata)

        todo.version += 1
        todo.updated_at = datetime.utcnow()

        # Save
        await redis.set(item_key, json.dumps(todo.to_dict()))

        # Update status index if changed
        if status is not None and old_status != status:
            old_status_key = self.STATUS_SET.format(user_id=todo.user_id, status=old_status.value)
            new_status_key = self.STATUS_SET.format(user_id=todo.user_id, status=status.value)
            await redis.srem(old_status_key, todo_id)
            await redis.sadd(new_status_key, todo_id)

        # Update priority index if changed
        if priority is not None and old_priority != priority:
            priority_key = self.BY_PRIORITY.format(user_id=todo.user_id)
            await redis.zadd(priority_key, {todo_id: priority.value})

        # Update tag indexes if changed
        if tags is not None:
            new_tags = set(tags)
            removed = old_tags - new_tags
            added = new_tags - old_tags
            for tag in removed:
                tag_key = self.TAG_SET.format(user_id=todo.user_id, tag=tag.lower())
                await redis.srem(tag_key, todo_id)
            for tag in added:
                tag_key = self.TAG_SET.format(user_id=todo.user_id, tag=tag.lower())
                await redis.sadd(tag_key, todo_id)

        # Update project index if changed
        if project_id is not None and old_project != project_id:
            if old_project:
                old_proj_key = self.PROJECT_SET.format(user_id=todo.user_id, project=old_project)
                await redis.srem(old_proj_key, todo_id)
            proj_key = self.PROJECT_SET.format(user_id=todo.user_id, project=project_id)
            await redis.sadd(proj_key, todo_id)

        # Update due date index
        due_key = self.BY_DUE.format(user_id=todo.user_id)
        if clear_due:
            await redis.zrem(due_key, todo_id)
        elif due_at is not None:
            await redis.zadd(due_key, {todo_id: due_at.timestamp()})

        await self._publish_event(todo.user_id, "updated", todo_id, todo.to_dict())

        logger.info(f"Updated todo {todo_id}")
        return todo

    async def delete(self, todo_id: str) -> bool:
        """Delete a todo and remove all indexes. Returns True if deleted."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(todo_id=todo_id)
        data = await redis.get(item_key)
        if not data:
            return False

        todo = TodoItem.from_dict(json.loads(data))

        # Remove from all index sets
        all_key = self.ALL_SET.format(user_id=todo.user_id)
        await redis.srem(all_key, todo_id)

        status_key = self.STATUS_SET.format(user_id=todo.user_id, status=todo.status.value)
        await redis.srem(status_key, todo_id)

        priority_key = self.BY_PRIORITY.format(user_id=todo.user_id)
        await redis.zrem(priority_key, todo_id)

        created_key = self.BY_CREATED.format(user_id=todo.user_id)
        await redis.zrem(created_key, todo_id)

        due_key = self.BY_DUE.format(user_id=todo.user_id)
        await redis.zrem(due_key, todo_id)

        if todo.project_id:
            proj_key = self.PROJECT_SET.format(user_id=todo.user_id, project=todo.project_id)
            await redis.srem(proj_key, todo_id)

        for tag in todo.tags:
            tag_key = self.TAG_SET.format(user_id=todo.user_id, tag=tag.lower())
            await redis.srem(tag_key, todo_id)

        # Remove subtask reference from parent
        if todo.parent_id:
            parent_subtasks = self.SUBTASKS_KEY.format(todo_id=todo.parent_id)
            await redis.lrem(parent_subtasks, 0, todo_id)

        # Remove subtask list
        subtask_key = self.SUBTASKS_KEY.format(todo_id=todo_id)
        await redis.delete(subtask_key)

        # Delete the item
        await redis.delete(item_key)

        await self._publish_event(todo.user_id, "deleted", todo_id)

        logger.info(f"Deleted todo {todo_id}")
        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def list_todos(
        self,
        user_id: str,
        status: Optional[TodoStatus] = None,
        priority: Optional[TodoPriority] = None,
        project_id: Optional[str] = None,
        tag: Optional[str] = None,
        due_before: Optional[datetime] = None,
        due_after: Optional[datetime] = None,
        sort_by: str = "priority",  # "priority", "due", "created"
        limit: int = 50,
    ) -> List[TodoItem]:
        """List todos with optional filters."""
        redis = await self._get_client()

        # Determine base set of IDs
        if status:
            base_key = self.STATUS_SET.format(user_id=user_id, status=status.value)
            todo_ids = await redis.smembers(base_key)
        elif project_id:
            proj_key = self.PROJECT_SET.format(user_id=user_id, project=project_id)
            todo_ids = await redis.smembers(proj_key)
        elif tag:
            tag_key = self.TAG_SET.format(user_id=user_id, tag=tag.lower())
            todo_ids = await redis.smembers(tag_key)
        elif due_before or due_after:
            due_key = self.BY_DUE.format(user_id=user_id)
            min_score = due_after.timestamp() if due_after else "-inf"
            max_score = due_before.timestamp() if due_before else "+inf"
            todo_ids = await redis.zrangebyscore(due_key, min_score, max_score)
        else:
            all_key = self.ALL_SET.format(user_id=user_id)
            todo_ids = await redis.smembers(all_key)

        if not todo_ids:
            return []

        # Batch fetch
        todos = await self.get_batch(list(todo_ids))

        # Apply additional filters
        if priority:
            todos = [t for t in todos if t.priority == priority]
        if status and not (project_id or tag):
            pass  # Already filtered by status set
        elif status:
            todos = [t for t in todos if t.status == status]
        if project_id and not (status is None and tag is None):
            todos = [t for t in todos if t.project_id == project_id]
        if tag and not (status is None and project_id is None):
            todos = [t for t in todos if tag.lower() in [t.lower() for t in t.tags]]
        if due_before and not (due_after is None):
            todos = [t for t in todos if t.due_at and t.due_at <= due_before]
        if due_after and not (due_before is None):
            todos = [t for t in todos if t.due_at and t.due_at >= due_after]

        # Sort
        if sort_by == "priority":
            todos.sort(key=lambda t: (-t.priority.value, t.due_at or datetime.max))
        elif sort_by == "due":
            todos.sort(key=lambda t: (t.due_at or datetime.max, -t.priority.value))
        elif sort_by == "created":
            todos.sort(key=lambda t: t.created_at, reverse=True)

        return todos[:limit]

    async def get_overdue(self, user_id: str) -> List[TodoItem]:
        """Get all overdue todos (past due, not completed/cancelled)."""
        redis = await self._get_client()
        due_key = self.BY_DUE.format(user_id=user_id)
        now_ts = datetime.utcnow().timestamp()

        todo_ids = await redis.zrangebyscore(due_key, "-inf", now_ts)
        if not todo_ids:
            return []

        todos = await self.get_batch(list(todo_ids))
        return [
            t for t in todos
            if t.status not in (TodoStatus.COMPLETED, TodoStatus.CANCELLED)
        ]

    async def get_subtasks(self, todo_id: str) -> List[TodoItem]:
        """Get all subtasks of a todo."""
        redis = await self._get_client()
        subtask_key = self.SUBTASKS_KEY.format(todo_id=todo_id)
        child_ids = await redis.lrange(subtask_key, 0, -1)
        if not child_ids:
            return []
        return await self.get_batch(child_ids)

    async def get_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of the user's todos for dashboard display."""
        redis = await self._get_client()

        # Count by status
        status_counts = {}
        for status in TodoStatus:
            key = self.STATUS_SET.format(user_id=user_id, status=status.value)
            count = await redis.scard(key)
            status_counts[status.value] = count

        # Count overdue
        overdue = await self.get_overdue(user_id)

        # Due today (within next 24h)
        now = datetime.utcnow()
        due_key = self.BY_DUE.format(user_id=user_id)
        end_of_day_ts = now.replace(hour=23, minute=59, second=59).timestamp()
        due_today_ids = await redis.zrangebyscore(due_key, now.timestamp(), end_of_day_ts)
        due_today = await self.get_batch(list(due_today_ids)) if due_today_ids else []
        due_today = [t for t in due_today if t.status not in (TodoStatus.COMPLETED, TodoStatus.CANCELLED)]

        # Total
        all_key = self.ALL_SET.format(user_id=user_id)
        total = await redis.scard(all_key)

        return {
            "total": total,
            "by_status": status_counts,
            "overdue_count": len(overdue),
            "due_today_count": len(due_today),
            "overdue": [t.format_for_display() for t in overdue[:5]],
            "due_today": [t.format_for_display() for t in due_today[:5]],
        }

    async def complete(self, todo_id: str) -> Optional[TodoItem]:
        """Mark a todo as completed. Convenience wrapper for update."""
        todo = await self.update(todo_id, status=TodoStatus.COMPLETED)
        if todo and self._completion_tracker:
            try:
                await self._completion_tracker.record_completion(todo)
            except Exception as e:
                logger.warning(f"Failed to record completion for adaptation: {e}")
        return todo

    async def reschedule(
        self, todo_id: str, new_due: Optional[datetime] = None, new_scheduled: Optional[datetime] = None
    ) -> Optional[TodoItem]:
        """Reschedule a todo, incrementing the reschedule counter."""
        todo = await self.get(todo_id)
        if not todo:
            return None

        redis = await self._get_client()
        todo.reschedule_count += 1
        if new_due:
            todo.due_at = new_due
        if new_scheduled:
            todo.scheduled_at = new_scheduled
        todo.version += 1
        todo.updated_at = datetime.utcnow()

        item_key = self.ITEM_KEY.format(todo_id=todo_id)
        await redis.set(item_key, json.dumps(todo.to_dict()))

        # Update due index
        if new_due:
            due_key = self.BY_DUE.format(user_id=todo.user_id)
            await redis.zadd(due_key, {todo_id: new_due.timestamp()})

        await self._publish_event(todo.user_id, "rescheduled", todo_id, todo.to_dict())

        # Track reschedule for adaptation
        if self._completion_tracker:
            try:
                old_due = todo.due_at if not new_due else (todo.due_at or datetime.utcnow())
                await self._completion_tracker.record_reschedule(
                    todo, from_time=old_due, to_time=new_due or new_scheduled or datetime.utcnow()
                )
            except Exception as e:
                logger.warning(f"Failed to record reschedule for adaptation: {e}")

        logger.info(f"Rescheduled todo {todo_id} (count: {todo.reschedule_count})")
        return todo
