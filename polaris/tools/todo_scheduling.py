"""Todo scheduling tools - link todos to Google Calendar events."""

import logging
from datetime import datetime
from typing import List

from shared.tools.interface import Tool
from shared.llm.types import ToolParameter
from polaris.tools.utils import PolarisToolContext

logger = logging.getLogger(__name__)


class ScheduleTodoTool(Tool):
    """Tool for scheduling a todo as a Google Calendar event."""

    @property
    def name(self) -> str:
        return "schedule_todo"

    @property
    def description(self) -> str:
        return (
            "Schedule a todo as a Google Calendar event. Creates a calendar event "
            "linked to the todo. Changes to the todo's time will update the event."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todo_id",
                type="string",
                description="The todo ID to schedule (e.g., 'todo-1')",
                required=True,
            ),
            ToolParameter(
                name="date",
                type="string",
                description="Date to schedule (e.g., 'tomorrow', '2025-02-15')",
                required=True,
            ),
            ToolParameter(
                name="time",
                type="string",
                description="Time to schedule (e.g., '2pm', '14:30')",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="Duration in minutes (default: uses todo's estimate or 30)",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available."
        if not context.calendar_sync_manager:
            return "Error: Calendar sync not available. Requires both Redis and Google Calendar."

        todo_id = kwargs.get("todo_id")
        if not todo_id:
            return "Error: todo_id is required."

        todo = await context.todo_manager.get(todo_id)
        if not todo:
            return f"Todo not found: {todo_id}"

        # Parse the scheduled datetime
        try:
            from polaris.tools.utils import parse_datetime
            tz = context.config.timezone if context.config else "UTC"
            scheduled_at = parse_datetime(kwargs["date"], kwargs.get("time"), timezone=tz)
        except ValueError as e:
            return f"Error parsing date/time: {e}"

        # Check if already scheduled
        existing = await context.calendar_sync_manager.get_event_id_for_todo(todo_id)
        if existing:
            # Update existing event
            if scheduled_at.tzinfo:
                scheduled_at_naive = scheduled_at.replace(tzinfo=None)
            else:
                scheduled_at_naive = scheduled_at
            await context.todo_manager.update(
                todo_id,
                scheduled_at=scheduled_at_naive,
                estimated_duration_minutes=kwargs.get("duration_minutes"),
            )
            # Refresh the todo object
            todo = await context.todo_manager.get(todo_id)
            success = await context.calendar_sync_manager.update_calendar_event(todo)
            if success:
                return f"Rescheduled todo **{todo.title}** to {scheduled_at.strftime('%B %d at %I:%M %p')}"
            else:
                return "Failed to update the calendar event."

        # Schedule new
        if scheduled_at.tzinfo:
            scheduled_at_naive = scheduled_at.replace(tzinfo=None)
        else:
            scheduled_at_naive = scheduled_at
        await context.todo_manager.update(
            todo_id,
            scheduled_at=scheduled_at_naive,
            estimated_duration_minutes=kwargs.get("duration_minutes"),
        )
        # Refresh
        todo = await context.todo_manager.get(todo_id)

        event_id = await context.calendar_sync_manager.schedule_todo(
            todo=todo,
            scheduled_at=scheduled_at,
            duration_minutes=kwargs.get("duration_minutes"),
        )

        if event_id:
            # Store the event ID on the todo
            await context.todo_manager.update(todo_id, calendar_event_id=event_id)
            return (
                f"Scheduled todo **{todo.title}** for {scheduled_at.strftime('%B %d at %I:%M %p')}\n"
                f"Calendar event created."
            )
        else:
            return (
                f"Todo time updated to {scheduled_at.strftime('%B %d at %I:%M %p')}, "
                f"but failed to create calendar event (is Google Calendar connected?)."
            )


class UnscheduleTodoTool(Tool):
    """Tool for removing a todo from the calendar."""

    @property
    def name(self) -> str:
        return "unschedule_todo"

    @property
    def description(self) -> str:
        return (
            "Remove a todo's linked Google Calendar event. "
            "The todo itself is kept, but the calendar event is deleted."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todo_id",
                type="string",
                description="The todo ID to unschedule (e.g., 'todo-1')",
                required=True,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available."
        if not context.calendar_sync_manager:
            return "Error: Calendar sync not available."

        todo_id = kwargs.get("todo_id")
        if not todo_id:
            return "Error: todo_id is required."

        todo = await context.todo_manager.get(todo_id)
        if not todo:
            return f"Todo not found: {todo_id}"

        success = await context.calendar_sync_manager.unschedule_todo(todo)
        if success:
            await context.todo_manager.update(
                todo_id, clear_scheduled=True, calendar_event_id=""
            )
            return f"Unscheduled todo **{todo.title}** - calendar event removed."
        else:
            return f"Todo **{todo.title}** doesn't have a linked calendar event."
