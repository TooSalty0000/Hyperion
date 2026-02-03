"""Smart adaptation engine for Polaris todo system.

Tracks user completion patterns and makes intelligent rescheduling decisions:
- Records completions, reschedules, and snoozes
- Computes stats: completion rate by priority, preferred work hours, etc.
- Auto-reschedules low-priority missed items
- Asks user for high-priority missed items

Redis keys:
- hyperion:adapt:user:{user_id}:completions   -> LIST of JSON records (capped 200)
- hyperion:adapt:user:{user_id}:reschedules   -> LIST of JSON records (capped 100)
- hyperion:adapt:user:{user_id}:stats         -> JSON computed stats (1h TTL cache)
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from polaris.todo.models import TodoItem, TodoStatus, TodoPriority

logger = logging.getLogger(__name__)

_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError("redis is required for AdaptationEngine.")
    return _redis


@dataclass
class CompletionRecord:
    """Record of a todo completion event."""
    todo_id: str
    priority: int
    tags: List[str]
    was_overdue: bool
    delay_hours: float  # hours past due (0 if on time, negative if early)
    estimated_minutes: Optional[int]
    actual_minutes: Optional[int]  # if tracked
    completed_at: str  # ISO format

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionRecord":
        return cls(**data)


@dataclass
class RescheduleRecord:
    """Record of a todo reschedule event."""
    todo_id: str
    priority: int
    reason: str  # "auto", "user", "overdue"
    from_time: str  # ISO
    to_time: str  # ISO
    reschedule_count: int
    rescheduled_at: str  # ISO

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RescheduleRecord":
        return cls(**data)


@dataclass
class UserPatternStats:
    """Computed statistics about a user's task completion patterns."""
    completion_rate: float = 0.0  # overall
    completion_rate_by_priority: Dict[str, float] = field(default_factory=dict)
    avg_delay_hours_by_priority: Dict[str, float] = field(default_factory=dict)
    preferred_hours: List[int] = field(default_factory=list)  # hours of day (0-23)
    preferred_days: List[int] = field(default_factory=list)  # weekdays (0=Mon, 6=Sun)
    avg_reschedule_count: float = 0.0
    estimation_accuracy: float = 1.0  # ratio of actual/estimated
    total_completed: int = 0
    total_rescheduled: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserPatternStats":
        return cls(**data)


class CompletionTracker:
    """Tracks completion and reschedule events."""

    COMPLETIONS_KEY = "hyperion:adapt:user:{user_id}:completions"
    RESCHEDULES_KEY = "hyperion:adapt:user:{user_id}:reschedules"
    MAX_COMPLETIONS = 200
    MAX_RESCHEDULES = 100

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

    async def record_completion(self, todo: TodoItem):
        """Record a todo completion for pattern tracking."""
        redis = await self._get_client()

        delay_hours = 0.0
        if todo.due_at:
            delta = (todo.completed_at or datetime.utcnow()) - todo.due_at
            delay_hours = delta.total_seconds() / 3600

        record = CompletionRecord(
            todo_id=todo.id,
            priority=todo.priority.value,
            tags=todo.tags,
            was_overdue=delay_hours > 0,
            delay_hours=round(delay_hours, 2),
            estimated_minutes=todo.estimated_duration_minutes,
            actual_minutes=None,
            completed_at=datetime.utcnow().isoformat(),
        )

        key = self.COMPLETIONS_KEY.format(user_id=todo.user_id)
        await redis.lpush(key, json.dumps(record.to_dict()))
        await redis.ltrim(key, 0, self.MAX_COMPLETIONS - 1)

    async def record_reschedule(
        self, todo: TodoItem, from_time: datetime, to_time: datetime, reason: str = "auto"
    ):
        """Record a reschedule event."""
        redis = await self._get_client()

        record = RescheduleRecord(
            todo_id=todo.id,
            priority=todo.priority.value,
            reason=reason,
            from_time=from_time.isoformat(),
            to_time=to_time.isoformat(),
            reschedule_count=todo.reschedule_count,
            rescheduled_at=datetime.utcnow().isoformat(),
        )

        key = self.RESCHEDULES_KEY.format(user_id=todo.user_id)
        await redis.lpush(key, json.dumps(record.to_dict()))
        await redis.ltrim(key, 0, self.MAX_RESCHEDULES - 1)

    async def get_completions(self, user_id: str, limit: int = 200) -> List[CompletionRecord]:
        """Get recent completion records."""
        redis = await self._get_client()
        key = self.COMPLETIONS_KEY.format(user_id=user_id)
        raw = await redis.lrange(key, 0, limit - 1)
        return [CompletionRecord.from_dict(json.loads(r)) for r in raw]

    async def get_reschedules(self, user_id: str, limit: int = 100) -> List[RescheduleRecord]:
        """Get recent reschedule records."""
        redis = await self._get_client()
        key = self.RESCHEDULES_KEY.format(user_id=user_id)
        raw = await redis.lrange(key, 0, limit - 1)
        return [RescheduleRecord.from_dict(json.loads(r)) for r in raw]


class AdaptationEngine:
    """
    Makes intelligent rescheduling decisions based on user patterns.

    Decision matrix:
    | Priority | Missed Behavior                                              |
    |----------|--------------------------------------------------------------|
    | CRITICAL | Always notify immediately, never auto-reschedule             |
    | HIGH     | Notify with suggested reschedule time, wait for response     |
    | MEDIUM   | Auto-reschedule if reschedule_count < 2, else ask user       |
    | LOW      | Auto-reschedule silently, ask after 3 reschedules            |
    """

    STATS_KEY = "hyperion:adapt:user:{user_id}:stats"
    STATS_TTL = 3600  # 1 hour cache

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None
        self.tracker = CompletionTracker(redis_url)

    async def _get_client(self):
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def compute_stats(self, user_id: str) -> UserPatternStats:
        """Compute user pattern stats from completion/reschedule history."""
        redis = await self._get_client()

        # Check cache
        cache_key = self.STATS_KEY.format(user_id=user_id)
        cached = await redis.get(cache_key)
        if cached:
            return UserPatternStats.from_dict(json.loads(cached))

        completions = await self.tracker.get_completions(user_id)
        reschedules = await self.tracker.get_reschedules(user_id)

        stats = UserPatternStats()
        stats.total_completed = len(completions)
        stats.total_rescheduled = len(reschedules)

        if completions:
            # Completion rate by priority
            priority_counts: Dict[str, int] = Counter()
            priority_on_time: Dict[str, int] = Counter()
            delay_sums: Dict[str, float] = Counter()
            hour_counts: Counter = Counter()
            day_counts: Counter = Counter()
            estimation_ratios = []

            for c in completions:
                p_key = str(c.priority)
                priority_counts[p_key] += 1
                if not c.was_overdue:
                    priority_on_time[p_key] += 1
                delay_sums[p_key] += c.delay_hours

                try:
                    dt = datetime.fromisoformat(c.completed_at)
                    hour_counts[dt.hour] += 1
                    day_counts[dt.weekday()] += 1
                except (ValueError, AttributeError):
                    pass

                if c.estimated_minutes and c.actual_minutes:
                    estimation_ratios.append(c.actual_minutes / c.estimated_minutes)

            # Overall completion rate (on-time)
            on_time_total = sum(priority_on_time.values())
            stats.completion_rate = round(on_time_total / len(completions), 3)

            # Per-priority
            for p, count in priority_counts.items():
                stats.completion_rate_by_priority[p] = round(
                    priority_on_time.get(p, 0) / count, 3
                )
                stats.avg_delay_hours_by_priority[p] = round(delay_sums[p] / count, 2)

            # Preferred hours (top 5)
            stats.preferred_hours = [h for h, _ in hour_counts.most_common(5)]

            # Preferred days (top 3)
            stats.preferred_days = [d for d, _ in day_counts.most_common(3)]

            # Estimation accuracy
            if estimation_ratios:
                stats.estimation_accuracy = round(
                    sum(estimation_ratios) / len(estimation_ratios), 3
                )

        if reschedules:
            reschedule_counts = [r.reschedule_count for r in reschedules]
            stats.avg_reschedule_count = round(
                sum(reschedule_counts) / len(reschedule_counts), 2
            )

        # Cache for 1 hour
        await redis.set(cache_key, json.dumps(stats.to_dict()), ex=self.STATS_TTL)

        return stats

    async def decide_overdue_action(self, todo: TodoItem) -> Dict[str, Any]:
        """
        Decide what to do with an overdue todo.

        Returns a dict with:
        - action: "notify" | "auto_reschedule" | "ask_user" | "deactivate_prompt"
        - suggested_time: Optional[datetime] — when to reschedule
        - message: str — explanation for the user
        """
        stats = await self.compute_stats(todo.user_id)

        priority = todo.priority
        reschedule_count = todo.reschedule_count

        # Suggest a reschedule time based on user patterns
        suggested_time = self._suggest_reschedule_time(stats)

        if priority == TodoPriority.CRITICAL:
            return {
                "action": "notify",
                "suggested_time": None,
                "message": f"OVERDUE critical task: {todo.title}",
            }

        if priority == TodoPriority.HIGH:
            return {
                "action": "ask_user",
                "suggested_time": suggested_time,
                "message": (
                    f"Overdue high-priority task: {todo.title}. "
                    f"Suggested reschedule: {suggested_time.strftime('%a %I:%M %p') if suggested_time else 'tomorrow'}"
                ),
            }

        if priority == TodoPriority.MEDIUM:
            if reschedule_count < 2:
                return {
                    "action": "auto_reschedule",
                    "suggested_time": suggested_time,
                    "message": f"Auto-rescheduled: {todo.title}",
                }
            else:
                return {
                    "action": "ask_user",
                    "suggested_time": suggested_time,
                    "message": (
                        f"Task '{todo.title}' has been rescheduled {reschedule_count} times. "
                        f"Still want to keep it?"
                    ),
                }

        # LOW priority
        if reschedule_count < 3:
            return {
                "action": "auto_reschedule",
                "suggested_time": suggested_time,
                "message": f"Auto-rescheduled low-priority: {todo.title}",
            }
        else:
            return {
                "action": "deactivate_prompt",
                "suggested_time": None,
                "message": (
                    f"Low-priority task '{todo.title}' has been rescheduled "
                    f"{reschedule_count} times. Want to keep, defer, or cancel?"
                ),
            }

    def _suggest_reschedule_time(self, stats: UserPatternStats) -> Optional[datetime]:
        """Suggest a reschedule time based on user's preferred hours/days."""
        now = datetime.utcnow()

        # Default: next day at 10am
        target_hour = 10
        if stats.preferred_hours:
            target_hour = stats.preferred_hours[0]

        # Start from tomorrow
        candidate = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        candidate += timedelta(days=1)

        # If user has preferred days, find the next one
        if stats.preferred_days:
            for _ in range(7):
                if candidate.weekday() in stats.preferred_days:
                    break
                candidate += timedelta(days=1)

        return candidate
