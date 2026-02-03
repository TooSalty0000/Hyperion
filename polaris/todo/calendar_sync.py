"""Bidirectional sync between todos and Google Calendar events."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Any

from polaris.todo.models import TodoItem, TodoStatus

logger = logging.getLogger(__name__)

_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError("redis is required for CalendarSyncManager.")
    return _redis


class CalendarSyncManager:
    """
    Manages bidirectional mapping between todos and Google Calendar events.

    Key patterns:
    - hyperion:todo:item:{todo_id}:calendar_event -> STRING: google_event_id
    - hyperion:todo:calendar_event:{event_id}     -> STRING: todo_id (reverse lookup)
    """

    TODO_EVENT_KEY = "hyperion:todo:item:{todo_id}:calendar_event"
    EVENT_TODO_KEY = "hyperion:todo:calendar_event:{event_id}"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        calendar_service: Any = None,
        config: Any = None,
    ):
        self._redis_url = redis_url
        self._redis = None
        self.calendar_service = calendar_service
        self.config = config

    async def _get_client(self):
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    def set_calendar_service(self, service: Any):
        """Update the calendar service (called after auth)."""
        self.calendar_service = service

    # =========================================================================
    # Link Management
    # =========================================================================

    async def _link(self, todo_id: str, event_id: str):
        """Create bidirectional link between todo and calendar event."""
        redis = await self._get_client()
        todo_key = self.TODO_EVENT_KEY.format(todo_id=todo_id)
        event_key = self.EVENT_TODO_KEY.format(event_id=event_id)
        await redis.set(todo_key, event_id)
        await redis.set(event_key, todo_id)

    async def _unlink(self, todo_id: str, event_id: str):
        """Remove bidirectional link."""
        redis = await self._get_client()
        todo_key = self.TODO_EVENT_KEY.format(todo_id=todo_id)
        event_key = self.EVENT_TODO_KEY.format(event_id=event_id)
        await redis.delete(todo_key, event_key)

    async def get_event_id_for_todo(self, todo_id: str) -> Optional[str]:
        """Get the linked Google Calendar event ID for a todo."""
        redis = await self._get_client()
        key = self.TODO_EVENT_KEY.format(todo_id=todo_id)
        return await redis.get(key)

    async def get_todo_id_for_event(self, event_id: str) -> Optional[str]:
        """Get the linked todo ID for a Google Calendar event."""
        redis = await self._get_client()
        key = self.EVENT_TODO_KEY.format(event_id=event_id)
        return await redis.get(key)

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def schedule_todo(
        self,
        todo: TodoItem,
        scheduled_at: datetime,
        duration_minutes: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create a Google Calendar event for a todo.

        Returns the created event ID, or None on failure.
        """
        if not self.calendar_service:
            logger.warning("Calendar service not available for todo scheduling")
            return None

        tz = self.config.timezone if self.config else "UTC"
        duration = duration_minutes or todo.estimated_duration_minutes or 30

        # Build event title with [Todo] prefix
        title = f"[Todo] {todo.title}"

        # Build description
        desc_parts = []
        if todo.description:
            desc_parts.append(todo.description)
        desc_parts.append(f"Todo ID: {todo.id}")
        desc_parts.append(f"Priority: {todo.priority.name}")
        if todo.tags:
            desc_parts.append(f"Tags: {', '.join(todo.tags)}")
        description = "\n".join(desc_parts)

        # Format datetime for API
        from polaris.tools.utils import format_datetime_for_api
        end_dt = scheduled_at + timedelta(minutes=duration)

        event_body = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": format_datetime_for_api(scheduled_at, timezone=tz),
                "timeZone": tz,
            },
            "end": {
                "dateTime": format_datetime_for_api(end_dt, timezone=tz),
                "timeZone": tz,
            },
        }

        try:
            calendar_id = self.config.default_calendar_id if self.config else "primary"
            created = (
                self.calendar_service.events()
                .insert(calendarId=calendar_id, body=event_body)
                .execute()
            )
            event_id = created["id"]

            # Create bidirectional link
            await self._link(todo.id, event_id)

            logger.info(f"Scheduled todo {todo.id} as calendar event {event_id[:8]}...")
            return event_id

        except Exception as e:
            logger.error(f"Failed to create calendar event for todo {todo.id}: {e}")
            return None

    async def unschedule_todo(self, todo: TodoItem) -> bool:
        """
        Remove the Google Calendar event linked to a todo.

        Returns True if successfully unscheduled.
        """
        event_id = await self.get_event_id_for_todo(todo.id)
        if not event_id:
            return False

        if self.calendar_service:
            try:
                calendar_id = self.config.default_calendar_id if self.config else "primary"
                self.calendar_service.events().delete(
                    calendarId=calendar_id, eventId=event_id
                ).execute()
            except Exception as e:
                # 410 = already deleted, that's fine
                if hasattr(e, 'resp') and e.resp.status == 410:
                    pass
                else:
                    logger.error(f"Failed to delete calendar event {event_id}: {e}")

        await self._unlink(todo.id, event_id)
        logger.info(f"Unscheduled todo {todo.id} (removed event {event_id[:8]}...)")
        return True

    async def update_calendar_event(self, todo: TodoItem) -> bool:
        """
        Update the linked calendar event to match the todo's current state.

        Called when a todo's scheduled_at, title, or description changes.
        """
        event_id = await self.get_event_id_for_todo(todo.id)
        if not event_id or not self.calendar_service:
            return False

        tz = self.config.timezone if self.config else "UTC"
        calendar_id = self.config.default_calendar_id if self.config else "primary"

        try:
            # Fetch current event
            event = (
                self.calendar_service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )

            # Update title
            event["summary"] = f"[Todo] {todo.title}"

            # Update description
            desc_parts = []
            if todo.description:
                desc_parts.append(todo.description)
            desc_parts.append(f"Todo ID: {todo.id}")
            desc_parts.append(f"Priority: {todo.priority.name}")
            if todo.tags:
                desc_parts.append(f"Tags: {', '.join(todo.tags)}")
            event["description"] = "\n".join(desc_parts)

            # Update time if scheduled_at changed
            if todo.scheduled_at:
                from polaris.tools.utils import format_datetime_for_api
                duration = todo.estimated_duration_minutes or 30
                end_dt = todo.scheduled_at + timedelta(minutes=duration)

                event["start"] = {
                    "dateTime": format_datetime_for_api(todo.scheduled_at, timezone=tz),
                    "timeZone": tz,
                }
                event["end"] = {
                    "dateTime": format_datetime_for_api(end_dt, timezone=tz),
                    "timeZone": tz,
                }

            self.calendar_service.events().update(
                calendarId=calendar_id, eventId=event_id, body=event
            ).execute()

            logger.info(f"Updated calendar event for todo {todo.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update calendar event for todo {todo.id}: {e}")
            return False

    async def is_todo_linked(self, event_id: str) -> bool:
        """Check if a calendar event is linked to a todo."""
        todo_id = await self.get_todo_id_for_event(event_id)
        return todo_id is not None
