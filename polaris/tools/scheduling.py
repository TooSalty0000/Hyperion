"""Scheduling tools for Polaris."""

import logging
from datetime import datetime, timedelta, time as dt_time
from typing import List, Tuple

from shared.tools.interface import Tool
from shared.llm.types import ToolParameter
from polaris.tools.utils import (
    PolarisToolContext,
    parse_datetime,
    format_datetime_for_api,
)

logger = logging.getLogger(__name__)


class FindFreeSlotsTool(Tool):
    """Tool for finding free time slots in the calendar."""

    @property
    def name(self) -> str:
        return "find_free_slots"

    @property
    def description(self) -> str:
        return (
            "Find available time slots for scheduling. Searches within working hours "
            "and returns slots of the requested duration. Useful for finding meeting times."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="date",
                type="string",
                description="Date to search for free slots (e.g., 'tomorrow', 'next Monday')",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="Required duration in minutes (default: 60)",
                required=False,
            ),
            ToolParameter(
                name="start_hour",
                type="integer",
                description="Earliest hour to consider (default: from config, usually 9)",
                required=False,
            ),
            ToolParameter(
                name="end_hour",
                type="integer",
                description="Latest hour to consider (default: from config, usually 17)",
                required=False,
            ),
            ToolParameter(
                name="calendar_id",
                type="string",
                description="Calendar ID (default: 'primary')",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        date_str = kwargs.get("date")
        duration_minutes = kwargs.get("duration_minutes", 60)
        calendar_id = kwargs.get("calendar_id", "primary")

        # Get working hours from config or params
        if context.config:
            start_hour = kwargs.get("start_hour", context.config.working_hours_start)
            end_hour = kwargs.get("end_hour", context.config.working_hours_end)
        else:
            start_hour = kwargs.get("start_hour", 9)
            end_hour = kwargs.get("end_hour", 17)

        tz = context.config.timezone if context.config else "UTC"

        if not context.calendar_service:
            return "Error: Google Calendar service not available."

        try:
            # Parse the target date
            target_date = parse_datetime(date_str, timezone=tz).date()

            # Define the search window
            day_start = datetime.combine(target_date, dt_time(start_hour, 0))
            day_end = datetime.combine(target_date, dt_time(end_hour, 0))

            # Get events for that day
            events_result = (
                context.calendar_service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=format_datetime_for_api(day_start, timezone=tz),
                    timeMax=format_datetime_for_api(day_end, timezone=tz),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            # Build list of busy periods
            busy_periods: List[Tuple[datetime, datetime]] = []
            for event in events:
                start = event.get("start", {})
                end = event.get("end", {})

                if "dateTime" in start and "dateTime" in end:
                    start_dt = datetime.fromisoformat(
                        start["dateTime"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    end_dt = datetime.fromisoformat(
                        end["dateTime"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    busy_periods.append((start_dt, end_dt))

            # Sort busy periods
            busy_periods.sort(key=lambda x: x[0])

            # Find free slots
            free_slots: List[Tuple[datetime, datetime]] = []
            current_time = day_start
            duration = timedelta(minutes=duration_minutes)

            for busy_start, busy_end in busy_periods:
                # Check if there's a gap before this busy period
                if current_time + duration <= busy_start:
                    free_slots.append((current_time, busy_start))
                # Move current time to after this busy period
                current_time = max(current_time, busy_end)

            # Check if there's time after the last event
            if current_time + duration <= day_end:
                free_slots.append((current_time, day_end))

            # Format output
            if not free_slots:
                return (
                    f"No free slots of {duration_minutes} minutes available on "
                    f"{target_date.strftime('%A, %B %d')} between "
                    f"{start_hour}:00 and {end_hour}:00."
                )

            day_name = target_date.strftime("%A")
            date_formatted = target_date.strftime("%B %d, %Y")

            lines = [
                f"**Free slots on {day_name}, {date_formatted}:**\n",
                f"_(Looking for {duration_minutes} minute slots)_\n",
            ]

            for slot_start, slot_end in free_slots:
                start_str = slot_start.strftime("%I:%M %p")
                end_str = slot_end.strftime("%I:%M %p")
                slot_duration = int((slot_end - slot_start).total_seconds() / 60)
                lines.append(f"- {start_str} - {end_str} ({slot_duration} min available)")

            return "\n".join(lines)

        except ValueError as e:
            return f"Error parsing date: {str(e)}"
        except Exception as e:
            logger.error(f"Error finding free slots: {e}", exc_info=True)
            return f"Error finding free slots: {str(e)}"


class CheckConflictsTool(Tool):
    """Tool for checking scheduling conflicts."""

    @property
    def name(self) -> str:
        return "check_conflicts"

    @property
    def description(self) -> str:
        return (
            "Check if a proposed time slot conflicts with existing calendar events. "
            "Use before creating events to ensure no overlaps."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="date",
                type="string",
                description="Date to check",
                required=True,
            ),
            ToolParameter(
                name="time",
                type="string",
                description="Start time to check",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="Duration in minutes (default: 60)",
                required=False,
            ),
            ToolParameter(
                name="calendar_id",
                type="string",
                description="Calendar ID (default: 'primary')",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        date_str = kwargs.get("date")
        time_str = kwargs.get("time")
        duration_minutes = kwargs.get("duration_minutes", 60)
        calendar_id = kwargs.get("calendar_id", "primary")
        tz = context.config.timezone if context.config else "UTC"

        if not context.calendar_service:
            return "Error: Google Calendar service not available."

        try:
            # Parse the proposed time
            proposed_start = parse_datetime(date_str, time_str, timezone=tz)
            proposed_end = proposed_start + timedelta(minutes=duration_minutes)

            # Search for events in a window around the proposed time
            window_start = proposed_start - timedelta(hours=2)
            window_end = proposed_end + timedelta(hours=2)

            events_result = (
                context.calendar_service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=format_datetime_for_api(window_start, timezone=tz),
                    timeMax=format_datetime_for_api(window_end, timezone=tz),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            # Check for conflicts
            conflicts = []
            for event in events:
                start = event.get("start", {})
                end = event.get("end", {})

                if "dateTime" in start and "dateTime" in end:
                    event_start = datetime.fromisoformat(
                        start["dateTime"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    event_end = datetime.fromisoformat(
                        end["dateTime"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                    # Check for overlap
                    if proposed_start < event_end and proposed_end > event_start:
                        conflicts.append(
                            {
                                "title": event.get("summary", "Untitled"),
                                "start": event_start,
                                "end": event_end,
                            }
                        )

            # Format output
            proposed_date = proposed_start.strftime("%A, %B %d")
            proposed_time = proposed_start.strftime("%I:%M %p")
            proposed_end_time = proposed_end.strftime("%I:%M %p")

            if not conflicts:
                return (
                    f"No conflicts found for {proposed_date} from "
                    f"{proposed_time} to {proposed_end_time}. "
                    f"This time slot is available!"
                )

            lines = [
                f"**Conflict detected!**\n",
                f"Proposed: {proposed_date}, {proposed_time} - {proposed_end_time}\n",
                f"Conflicts with:\n",
            ]

            for conflict in conflicts:
                start_str = conflict["start"].strftime("%I:%M %p")
                end_str = conflict["end"].strftime("%I:%M %p")
                lines.append(f"- **{conflict['title']}** ({start_str} - {end_str})")

            return "\n".join(lines)

        except ValueError as e:
            return f"Error parsing date/time: {str(e)}"
        except Exception as e:
            logger.error(f"Error checking conflicts: {e}", exc_info=True)
            return f"Error checking conflicts: {str(e)}"
