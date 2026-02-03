"""Reminder tools for Polaris."""

import logging
from datetime import datetime, timedelta
from typing import List

from shared.tools.interface import Tool
from shared.llm.types import ToolParameter
from polaris.tools.utils import PolarisToolContext
from polaris.todo.reminder_models import ReminderType

logger = logging.getLogger(__name__)


class SetReminderTool(Tool):
    """Tool for setting a reminder."""

    @property
    def name(self) -> str:
        return "set_reminder"

    @property
    def description(self) -> str:
        return (
            "Set a reminder for a specific time. Can be linked to a todo or standalone. "
            "The reminder will send a Discord DM when it fires."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type="string",
                description="What to remind about",
                required=True,
            ),
            ToolParameter(
                name="date",
                type="string",
                description="When to remind (e.g., 'tomorrow', '2025-02-15', 'in 2 hours')",
                required=True,
            ),
            ToolParameter(
                name="time",
                type="string",
                description="Time for the reminder (e.g., '9am', '14:30'). Not needed for relative times like 'in 2 hours'.",
                required=False,
            ),
            ToolParameter(
                name="todo_id",
                type="string",
                description="Link this reminder to a todo (optional)",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.reminder_manager:
            return "Error: Reminder system not available. Redis is required."

        message = kwargs.get("message")
        if not message:
            return "Error: message is required."

        date_str = kwargs.get("date", "")

        # Handle relative time (e.g., "in 2 hours", "in 30 minutes")
        fire_at = None
        date_lower = date_str.lower().strip()
        if date_lower.startswith("in "):
            parts = date_lower[3:].strip().split()
            if len(parts) >= 2:
                try:
                    amount = int(parts[0])
                    unit = parts[1]
                    if "hour" in unit:
                        fire_at = datetime.utcnow() + timedelta(hours=amount)
                    elif "min" in unit:
                        fire_at = datetime.utcnow() + timedelta(minutes=amount)
                    elif "day" in unit:
                        fire_at = datetime.utcnow() + timedelta(days=amount)
                except (ValueError, IndexError):
                    pass

        if not fire_at:
            try:
                from polaris.tools.utils import parse_datetime
                tz = context.config.timezone if context.config else "UTC"
                fire_at = parse_datetime(date_str, kwargs.get("time"), timezone=tz)
                if fire_at.tzinfo:
                    fire_at = fire_at.replace(tzinfo=None)
            except ValueError as e:
                return f"Error parsing date/time: {e}"

        user_id = str(context.user_id) if context.user_id else "default"

        try:
            reminder = await context.reminder_manager.create(
                user_id=user_id,
                message=message,
                fire_at=fire_at,
                todo_id=kwargs.get("todo_id"),
                channel_id=context.current_channel_id,
            )

            # If linked to a todo, add reminder ID to the todo
            if kwargs.get("todo_id") and context.todo_manager:
                todo = await context.todo_manager.get(kwargs["todo_id"])
                if todo:
                    todo.reminder_ids.append(reminder.id)
                    await context.todo_manager.update(
                        kwargs["todo_id"],
                        metadata={"reminder_ids": todo.reminder_ids},
                    )

            return (
                f"Reminder set: **{message}**\n"
                f"- Fire at: {fire_at.strftime('%B %d, %Y %I:%M %p')}\n"
                f"- ID: {reminder.id}"
            )

        except Exception as e:
            logger.error(f"Error setting reminder: {e}", exc_info=True)
            return f"Error setting reminder: {e}"


class ListRemindersTool(Tool):
    """Tool for listing pending reminders."""

    @property
    def name(self) -> str:
        return "list_reminders"

    @property
    def description(self) -> str:
        return "List all pending reminders, sorted by fire time."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.reminder_manager:
            return "Error: Reminder system not available."

        user_id = str(context.user_id) if context.user_id else "default"

        try:
            reminders = await context.reminder_manager.list_pending(user_id=user_id)

            if not reminders:
                return "No pending reminders."

            lines = [f"**Pending Reminders** ({len(reminders)})\n"]
            for r in reminders:
                todo_str = f" (todo: {r.todo_id})" if r.todo_id else ""
                lines.append(
                    f"- **{r.message}** — {r.fire_at.strftime('%b %d %I:%M %p')}"
                    f"{todo_str} [id: {r.id}]"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error listing reminders: {e}", exc_info=True)
            return f"Error listing reminders: {e}"


class SnoozeReminderTool(Tool):
    """Tool for snoozing a reminder."""

    @property
    def name(self) -> str:
        return "snooze_reminder"

    @property
    def description(self) -> str:
        return "Snooze a reminder for a specified number of minutes."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id",
                type="string",
                description="The reminder ID to snooze",
                required=True,
            ),
            ToolParameter(
                name="minutes",
                type="integer",
                description="Minutes to snooze (default: 15)",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.reminder_manager:
            return "Error: Reminder system not available."

        reminder_id = kwargs.get("reminder_id")
        minutes = kwargs.get("minutes", 15)

        try:
            reminder = await context.reminder_manager.snooze(reminder_id, minutes)
            if not reminder:
                return f"Reminder not found: {reminder_id}"

            return (
                f"Snoozed: **{reminder.message}**\n"
                f"New fire time: {reminder.fire_at.strftime('%B %d %I:%M %p')}"
            )

        except Exception as e:
            logger.error(f"Error snoozing reminder: {e}", exc_info=True)
            return f"Error snoozing reminder: {e}"


class DismissReminderTool(Tool):
    """Tool for dismissing a reminder."""

    @property
    def name(self) -> str:
        return "dismiss_reminder"

    @property
    def description(self) -> str:
        return "Dismiss (cancel) a pending reminder."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id",
                type="string",
                description="The reminder ID to dismiss",
                required=True,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.reminder_manager:
            return "Error: Reminder system not available."

        reminder_id = kwargs.get("reminder_id")

        try:
            success = await context.reminder_manager.dismiss(reminder_id)
            if success:
                return f"Reminder dismissed: {reminder_id}"
            else:
                return f"Reminder not found: {reminder_id}"

        except Exception as e:
            logger.error(f"Error dismissing reminder: {e}", exc_info=True)
            return f"Error dismissing reminder: {e}"
