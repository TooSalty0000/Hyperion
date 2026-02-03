"""Redis-backed reminder manager for Polaris."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any

from polaris.todo.reminder_models import ReminderItem, ReminderStatus, ReminderType

logger = logging.getLogger(__name__)

_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError("redis is required for ReminderManager.")
    return _redis


class ReminderManager:
    """
    Redis-backed reminder management.

    Key patterns:
    - hyperion:reminder:item:{reminder_id}   -> JSON ReminderItem
    - hyperion:reminder:pending              -> ZSET: reminder_id -> fire_at_timestamp
    - hyperion:reminder:todo:{todo_id}       -> SET of reminder_ids
    - hyperion:reminder:counter              -> INT
    """

    ITEM_KEY = "hyperion:reminder:item:{reminder_id}"
    PENDING_ZSET = "hyperion:reminder:pending"
    TODO_REMINDERS = "hyperion:reminder:todo:{todo_id}"
    USER_ALL = "hyperion:reminder:user:{user_id}:all"
    COUNTER_KEY = "hyperion:reminder:counter"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None

    async def _get_client(self):
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def _next_id(self) -> str:
        redis = await self._get_client()
        counter = await redis.incr(self.COUNTER_KEY)
        return f"rem-{counter}"

    # =========================================================================
    # CRUD
    # =========================================================================

    async def create(
        self,
        user_id: str,
        message: str,
        fire_at: datetime,
        todo_id: Optional[str] = None,
        reminder_types: Optional[List[ReminderType]] = None,
        channel_id: Optional[int] = None,
    ) -> ReminderItem:
        """Create a new reminder."""
        redis = await self._get_client()
        reminder_id = await self._next_id()

        reminder = ReminderItem(
            id=reminder_id,
            user_id=user_id,
            todo_id=todo_id,
            message=message,
            fire_at=fire_at,
            reminder_types=reminder_types or [ReminderType.CHANNEL_MENTION, ReminderType.WEB_PUSH],
            channel_id=channel_id,
        )

        # Store item
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        await redis.set(item_key, json.dumps(reminder.to_dict()))

        # Add to pending sorted set
        await redis.zadd(self.PENDING_ZSET, {reminder_id: fire_at.timestamp()})

        # Track per-user
        user_key = self.USER_ALL.format(user_id=user_id)
        await redis.sadd(user_key, reminder_id)

        # Link to todo if provided
        if todo_id:
            todo_key = self.TODO_REMINDERS.format(todo_id=todo_id)
            await redis.sadd(todo_key, reminder_id)

        logger.info(f"Created reminder {reminder_id} for {fire_at.isoformat()}")
        return reminder

    async def get(self, reminder_id: str) -> Optional[ReminderItem]:
        """Get a reminder by ID."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        data = await redis.get(item_key)
        if not data:
            return None
        return ReminderItem.from_dict(json.loads(data))

    async def get_for_todo(self, todo_id: str) -> List[ReminderItem]:
        """Get all reminders for a specific todo."""
        redis = await self._get_client()
        todo_key = self.TODO_REMINDERS.format(todo_id=todo_id)
        reminder_ids = await redis.smembers(todo_key)
        if not reminder_ids:
            return []

        reminders = []
        keys = [self.ITEM_KEY.format(reminder_id=rid) for rid in reminder_ids]
        results = await redis.mget(keys)
        for data in results:
            if data:
                reminders.append(ReminderItem.from_dict(json.loads(data)))
        return reminders

    async def dismiss(self, reminder_id: str) -> bool:
        """Dismiss a reminder (remove from pending)."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        data = await redis.get(item_key)
        if not data:
            return False

        reminder = ReminderItem.from_dict(json.loads(data))
        reminder.status = ReminderStatus.DISMISSED

        await redis.set(item_key, json.dumps(reminder.to_dict()))
        await redis.zrem(self.PENDING_ZSET, reminder_id)

        logger.info(f"Dismissed reminder {reminder_id}")
        return True

    async def snooze(self, reminder_id: str, minutes: int = 15) -> Optional[ReminderItem]:
        """Snooze a reminder by pushing its fire time forward."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        data = await redis.get(item_key)
        if not data:
            return None

        reminder = ReminderItem.from_dict(json.loads(data))
        reminder.fire_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        reminder.status = ReminderStatus.SNOOZED
        reminder.snooze_count += 1

        await redis.set(item_key, json.dumps(reminder.to_dict()))
        await redis.zadd(self.PENDING_ZSET, {reminder_id: reminder.fire_at.timestamp()})

        logger.info(f"Snoozed reminder {reminder_id} for {minutes} minutes")
        return reminder

    async def delete(self, reminder_id: str) -> bool:
        """Delete a reminder entirely."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        data = await redis.get(item_key)
        if not data:
            return False

        reminder = ReminderItem.from_dict(json.loads(data))

        # Remove from pending
        await redis.zrem(self.PENDING_ZSET, reminder_id)

        # Remove from todo link
        if reminder.todo_id:
            todo_key = self.TODO_REMINDERS.format(todo_id=reminder.todo_id)
            await redis.srem(todo_key, reminder_id)

        # Delete item
        await redis.delete(item_key)

        logger.info(f"Deleted reminder {reminder_id}")
        return True

    # =========================================================================
    # Firing
    # =========================================================================

    async def get_due_reminders(self) -> List[ReminderItem]:
        """
        Get all reminders that should fire now.

        Returns reminders where fire_at <= now, sorted by fire time.
        """
        redis = await self._get_client()
        now_ts = time.time()

        # Get all reminder IDs due now or earlier
        reminder_ids = await redis.zrangebyscore(self.PENDING_ZSET, "-inf", now_ts)
        if not reminder_ids:
            return []

        # Batch fetch
        keys = [self.ITEM_KEY.format(reminder_id=rid) for rid in reminder_ids]
        results = await redis.mget(keys)

        reminders = []
        for data in results:
            if data:
                r = ReminderItem.from_dict(json.loads(data))
                if r.status in (ReminderStatus.PENDING, ReminderStatus.SNOOZED):
                    reminders.append(r)

        return reminders

    async def mark_fired(self, reminder_id: str):
        """Mark a reminder as fired and remove from pending queue."""
        redis = await self._get_client()
        item_key = self.ITEM_KEY.format(reminder_id=reminder_id)
        data = await redis.get(item_key)
        if not data:
            return

        reminder = ReminderItem.from_dict(json.loads(data))
        reminder.status = ReminderStatus.FIRED
        reminder.fired_at = datetime.now(timezone.utc)

        await redis.set(item_key, json.dumps(reminder.to_dict()))
        await redis.zrem(self.PENDING_ZSET, reminder_id)

    async def list_pending(self, user_id: Optional[str] = None, limit: int = 20) -> List[ReminderItem]:
        """List pending reminders, optionally filtered by user."""
        redis = await self._get_client()
        reminder_ids = await redis.zrange(self.PENDING_ZSET, 0, -1)
        if not reminder_ids:
            return []

        keys = [self.ITEM_KEY.format(reminder_id=rid) for rid in reminder_ids]
        results = await redis.mget(keys)

        reminders = []
        for data in results:
            if data:
                r = ReminderItem.from_dict(json.loads(data))
                if user_id and r.user_id != user_id:
                    continue
                reminders.append(r)

        reminders.sort(key=lambda r: r.fire_at)
        return reminders[:limit]

    async def list_all(self, user_id: str, limit: int = 50) -> List[ReminderItem]:
        """List all reminders (pending, fired, snoozed — excludes dismissed).

        Scans all reminder item keys in Redis. Single-user system so user_id
        filtering is skipped — all reminders belong to the owner regardless
        of whether they were created via API ('owner') or Discord (bot ID).
        """
        redis = await self._get_client()

        # IDs are sequential (rem-1, rem-2, ...) — read counter and batch fetch
        counter = await redis.get(self.COUNTER_KEY)
        if not counter:
            return []

        max_id = int(counter)
        keys = [self.ITEM_KEY.format(reminder_id=f"rem-{i}") for i in range(1, max_id + 1)]
        results = await redis.mget(keys)

        reminders = []
        for data in results:
            if not data:
                continue
            try:
                r = ReminderItem.from_dict(json.loads(data))
                if r.status == ReminderStatus.DISMISSED:
                    continue
                reminders.append(r)
            except Exception:
                continue

        # Normalize all datetimes to naive UTC for safe comparison
        # (old data uses naive utcnow(), new data uses aware now(utc))
        def _to_ts(dt: datetime) -> float:
            if dt is None:
                return 0.0
            if dt.tzinfo is not None:
                return dt.timestamp()
            # Naive datetime assumed UTC
            return dt.replace(tzinfo=timezone.utc).timestamp()

        active = sorted([r for r in reminders if r.status != ReminderStatus.FIRED], key=lambda r: _to_ts(r.fire_at))
        fired = sorted([r for r in reminders if r.status == ReminderStatus.FIRED], key=lambda r: _to_ts(r.fired_at) or _to_ts(r.fire_at), reverse=True)
        return (active + fired)[:limit]
