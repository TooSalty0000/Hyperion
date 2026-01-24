"""Calendar operation tools for Polaris."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from shared.tools.interface import Tool
from shared.llm.types import ToolParameter
from polaris.tools.utils import (
    PolarisToolContext,
    parse_datetime,
    format_event_for_display,
    format_datetime_for_api,
)

logger = logging.getLogger(__name__)


async def _find_event_by_title(
    context: PolarisToolContext,
    title: str,
    event_date_str: Optional[str],
    calendar_id: str,
    timezone: str,
) -> Optional[str]:
    """
    Search for an event by title via the Google Calendar API.

    Queries the API directly (not cache) for reliability.
    Returns the event_id if found, None otherwise.
    """
    if not context.calendar_service:
        return None

    # Determine search window
    try:
        if event_date_str:
            search_date = parse_datetime(event_date_str, timezone=timezone)
        else:
            # Default: search today ± 1 day for flexibility
            try:
                import zoneinfo
                now = datetime.now(zoneinfo.ZoneInfo(timezone)).replace(tzinfo=None)
            except (ImportError, KeyError):
                now = datetime.now()
            search_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        search_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Search a 3-day window (yesterday through tomorrow) for robustness
    start_dt = search_date - timedelta(days=1)
    end_dt = search_date + timedelta(days=2)

    try:
        import zoneinfo
        tz_info = zoneinfo.ZoneInfo(timezone)
        api_start = start_dt.replace(tzinfo=tz_info)
        api_end = end_dt.replace(tzinfo=tz_info)
    except (ImportError, KeyError):
        from datetime import timezone as dt_tz
        api_start = start_dt.replace(tzinfo=dt_tz.utc)
        api_end = end_dt.replace(tzinfo=dt_tz.utc)

    try:
        events_result = (
            context.calendar_service.events()
            .list(
                calendarId=calendar_id,
                timeMin=api_start.isoformat(),
                timeMax=api_end.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
                q=title,  # Google's built-in text search
            )
            .execute()
        )

        events = events_result.get("items", [])

        # Try exact match first, then substring
        title_lower = title.lower()
        for event in events:
            if event.get("summary", "").lower() == title_lower:
                logger.info(f"[FindEvent] Exact match: '{event.get('summary')}' (id: {event['id'][:8]}...)")
                return event["id"]

        # Substring match
        for event in events:
            if title_lower in event.get("summary", "").lower():
                logger.info(f"[FindEvent] Substring match: '{event.get('summary')}' (id: {event['id'][:8]}...)")
                return event["id"]

        # Word overlap match (60%+ of query words in title)
        query_words = set(title_lower.split())
        for event in events:
            event_words = set(event.get("summary", "").lower().split())
            if query_words and len(query_words & event_words) / len(query_words) >= 0.6:
                logger.info(f"[FindEvent] Fuzzy match: '{event.get('summary')}' (id: {event['id'][:8]}...)")
                return event["id"]

        logger.warning(f"[FindEvent] No match for '{title}' among {len(events)} events")
        return None

    except Exception as e:
        logger.error(f"[FindEvent] API search failed: {e}")
        return None


class CreateEventTool(Tool):
    """Tool for creating calendar events."""

    @property
    def name(self) -> str:
        return "create_event"

    @property
    def description(self) -> str:
        return (
            "Create a new calendar event. Specify title, date, time, and optionally "
            "duration, location, and description. Use natural language for dates "
            "(e.g., 'tomorrow', 'next Monday') and times (e.g., '2pm', '14:30')."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="title",
                type="string",
                description="Event title/summary",
                required=True,
            ),
            ToolParameter(
                name="date",
                type="string",
                description="Date of the event (e.g., '2024-01-15', 'tomorrow', 'next Monday')",
                required=True,
            ),
            ToolParameter(
                name="time",
                type="string",
                description="Start time (e.g., '2pm', '14:30', '2:30 PM')",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="Duration in minutes (default: 60)",
                required=False,
            ),
            ToolParameter(
                name="location",
                type="string",
                description="Event location (optional)",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Event description/notes (optional)",
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
        title = kwargs.get("title")
        date = kwargs.get("date")
        time = kwargs.get("time")
        duration_minutes = kwargs.get("duration_minutes", 60)
        location = kwargs.get("location")
        description = kwargs.get("description")
        calendar_id = kwargs.get("calendar_id", "primary")

        if not context.calendar_service:
            return "Error: Google Calendar service not available. Please configure credentials."

        tz = context.config.timezone if context.config else "UTC"
        try:
            # Parse the datetime in the user's timezone
            start_dt = parse_datetime(date, time, timezone=tz)
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            # Build event body with timezone-aware datetimes
            event = {
                "summary": title,
                "start": {
                    "dateTime": format_datetime_for_api(start_dt, timezone=tz),
                    "timeZone": tz,
                },
                "end": {
                    "dateTime": format_datetime_for_api(end_dt, timezone=tz),
                    "timeZone": tz,
                },
            }

            if location:
                event["location"] = location
            if description:
                event["description"] = description

            # Create the event
            created_event = (
                context.calendar_service.events()
                .insert(calendarId=calendar_id, body=event)
                .execute()
            )

            # Format response
            day_name = start_dt.strftime("%A")
            date_str = start_dt.strftime("%B %d, %Y")
            time_str = start_dt.strftime("%I:%M %p")
            end_time_str = end_dt.strftime("%I:%M %p")

            response = (
                f"Event created successfully!\n\n"
                f"**{title}**\n"
                f"- Date: {day_name}, {date_str}\n"
                f"- Time: {time_str} - {end_time_str}\n"
                f"- Duration: {duration_minutes} minutes"
            )

            if location:
                response += f"\n- Location: {location}"

            event_link = created_event.get("htmlLink", "")
            if event_link:
                response += f"\n\n[View in Calendar]({event_link})"

            return response

        except ValueError as e:
            return f"Error parsing date/time: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating event: {e}", exc_info=True)
            return f"Error creating event: {str(e)}"


class ListEventsTool(Tool):
    """Tool for listing calendar events."""

    @property
    def name(self) -> str:
        return "list_events"

    @property
    def description(self) -> str:
        return (
            "List upcoming calendar events. Returns title, time, location, and event_id for each event. "
            "IMPORTANT: Use the event_id from this output when calling update_event or delete_event."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="start_date",
                type="string",
                description="Start date for listing (default: today)",
                required=False,
            ),
            ToolParameter(
                name="end_date",
                type="string",
                description="End date for listing (default: 7 days from start)",
                required=False,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of events to return (default: 10)",
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
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        max_results = kwargs.get("max_results", 10)
        calendar_id = kwargs.get("calendar_id", "primary")
        tz = context.config.timezone if context.config else "UTC"

        if not context.calendar_service:
            return "Error: Google Calendar service not available. Please configure credentials."

        try:
            # Parse dates in the user's timezone
            if start_date:
                start_dt = parse_datetime(start_date, timezone=tz)
            else:
                try:
                    import zoneinfo
                    now = datetime.now(zoneinfo.ZoneInfo(tz)).replace(tzinfo=None)
                except (ImportError, KeyError):
                    now = datetime.now()
                start_dt = now.replace(hour=0, minute=0, second=0)

            if end_date:
                end_dt = parse_datetime(end_date, timezone=tz)
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            else:
                end_dt = start_dt + timedelta(days=7)

            # Make timezone-aware for the API query (requires RFC3339)
            try:
                import zoneinfo
                tz_info = zoneinfo.ZoneInfo(tz)
                api_start = start_dt.replace(tzinfo=tz_info)
                api_end = end_dt.replace(tzinfo=tz_info)
            except (ImportError, KeyError):
                from datetime import timezone as dt_tz
                api_start = start_dt.replace(tzinfo=dt_tz.utc)
                api_end = end_dt.replace(tzinfo=dt_tz.utc)

            # Query events
            events_result = (
                context.calendar_service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=api_start.isoformat(),
                    timeMax=api_end.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            # Cache the fetched events for later lookup (update/delete operations)
            if context.calendar_cache and events:
                context.calendar_cache.cache_events(
                    events=events,
                    calendar_id=calendar_id,
                    start_date=start_dt,
                    end_date=end_dt,
                )

            if not events:
                return f"No events found between {start_dt.strftime('%B %d')} and {end_dt.strftime('%B %d, %Y')}."

            # Group events by date
            events_by_date = {}
            for event in events:
                start = event.get("start", {})
                if "dateTime" in start:
                    event_date = datetime.fromisoformat(
                        start["dateTime"].replace("Z", "+00:00")
                    ).date()
                else:
                    event_date = datetime.fromisoformat(start.get("date")).date()

                if event_date not in events_by_date:
                    events_by_date[event_date] = []
                events_by_date[event_date].append(event)

            # Format output
            lines = [f"**Upcoming Events** ({len(events)} total)\n"]

            for date in sorted(events_by_date.keys()):
                day_name = date.strftime("%A")
                date_str = date.strftime("%B %d")
                lines.append(f"\n**{day_name}, {date_str}:**")

                for event in events_by_date[date]:
                    lines.append(format_event_for_display(event))

            return "\n".join(lines)

        except ValueError as e:
            return f"Error parsing date: {str(e)}"
        except Exception as e:
            logger.error(f"Error listing events: {e}", exc_info=True)
            return f"Error listing events: {str(e)}"


class UpdateEventTool(Tool):
    """Tool for updating calendar events."""

    @property
    def name(self) -> str:
        return "update_event"

    @property
    def description(self) -> str:
        return (
            "Update an existing calendar event. PREFERRED: use event_id from list_events output. "
            "Alternatively, provide event_title for fuzzy search. "
            "Can update title, date, time, duration, location, or description."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="event_id",
                type="string",
                description="The event ID (from list_events output). PREFERRED method - most reliable.",
                required=False,
            ),
            ToolParameter(
                name="event_title",
                type="string",
                description="Title of the event to find (fallback if event_id not available). Searches via API.",
                required=False,
            ),
            ToolParameter(
                name="event_date",
                type="string",
                description="Date to narrow title search (e.g. 'today', '2025-01-24')",
                required=False,
            ),
            ToolParameter(
                name="title",
                type="string",
                description="New event title (optional)",
                required=False,
            ),
            ToolParameter(
                name="date",
                type="string",
                description="New date (optional)",
                required=False,
            ),
            ToolParameter(
                name="time",
                type="string",
                description="New start time (optional)",
                required=False,
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="New duration in minutes (optional)",
                required=False,
            ),
            ToolParameter(
                name="location",
                type="string",
                description="New location (optional)",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="New description (optional)",
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
        event_id = kwargs.get("event_id")
        event_title = kwargs.get("event_title")
        event_date_str = kwargs.get("event_date")
        calendar_id = kwargs.get("calendar_id", "primary")
        tz = context.config.timezone if context.config else "UTC"

        if not context.calendar_service:
            return "Error: Google Calendar service not available."

        # Resolve event_id by searching the API if not provided
        if not event_id and event_title:
            event_id = await _find_event_by_title(
                context, event_title, event_date_str, calendar_id, tz
            )
            if not event_id:
                return (
                    f"Could not find event matching '{event_title}'. "
                    f"Use list_events first to see available events with their event_id, "
                    f"then call update_event with the exact event_id."
                )

        if not event_id:
            return "Error: Please provide either event_id or event_title to identify the event."

        try:
            # Get the existing event
            event = (
                context.calendar_service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )

            # Update fields if provided
            if kwargs.get("title"):
                event["summary"] = kwargs["title"]

            if kwargs.get("location"):
                event["location"] = kwargs["location"]

            if kwargs.get("description"):
                event["description"] = kwargs["description"]

            # Handle date/time updates
            if kwargs.get("date") or kwargs.get("time"):
                # Get existing start time
                existing_start = event.get("start", {})
                if "dateTime" in existing_start:
                    existing_dt = datetime.fromisoformat(
                        existing_start["dateTime"].replace("Z", "+00:00")
                    )
                else:
                    existing_dt = datetime.now()

                # Parse new date/time
                new_date = kwargs.get("date")
                new_time = kwargs.get("time")

                if new_date and new_time:
                    new_start = parse_datetime(new_date, new_time, timezone=tz)
                elif new_date:
                    new_start = parse_datetime(new_date, timezone=tz)
                    new_start = new_start.replace(
                        hour=existing_dt.hour, minute=existing_dt.minute
                    )
                else:
                    new_start = parse_datetime("today", new_time, timezone=tz)
                    new_start = new_start.replace(
                        year=existing_dt.year,
                        month=existing_dt.month,
                        day=existing_dt.day,
                    )

                # Calculate duration
                existing_end = event.get("end", {})
                if "dateTime" in existing_end:
                    existing_end_dt = datetime.fromisoformat(
                        existing_end["dateTime"].replace("Z", "+00:00")
                    )
                    duration = existing_end_dt - existing_dt
                else:
                    duration = timedelta(hours=1)

                # Override duration if specified
                if kwargs.get("duration_minutes"):
                    duration = timedelta(minutes=kwargs["duration_minutes"])

                new_end = new_start + duration

                event["start"] = {
                    "dateTime": format_datetime_for_api(new_start, timezone=tz),
                    "timeZone": tz,
                }
                event["end"] = {
                    "dateTime": format_datetime_for_api(new_end, timezone=tz),
                    "timeZone": tz,
                }

            # Update the event
            updated_event = (
                context.calendar_service.events()
                .update(calendarId=calendar_id, eventId=event_id, body=event)
                .execute()
            )

            return f"Event updated successfully: **{updated_event.get('summary')}**"

        except Exception as e:
            logger.error(f"Error updating event: {e}", exc_info=True)
            return f"Error updating event: {str(e)}"


class DeleteEventTool(Tool):
    """Tool for deleting calendar events."""

    @property
    def name(self) -> str:
        return "delete_event"

    @property
    def description(self) -> str:
        return (
            "Delete/cancel a calendar event. You can specify the event by ID, "
            "or by title (and optionally date) to look it up from recently fetched events."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="event_id",
                type="string",
                description="The event ID to delete (optional if event_title provided)",
                required=False,
            ),
            ToolParameter(
                name="event_title",
                type="string",
                description="Title of the event to find and delete (used if event_id not provided)",
                required=False,
            ),
            ToolParameter(
                name="event_date",
                type="string",
                description="Date of the event to find (helps narrow down search when using event_title)",
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
        event_id = kwargs.get("event_id")
        event_title = kwargs.get("event_title")
        event_date_str = kwargs.get("event_date")
        calendar_id = kwargs.get("calendar_id", "primary")
        tz = context.config.timezone if context.config else "UTC"

        if not context.calendar_service:
            return "Error: Google Calendar service not available."

        # Resolve event_id by searching the API if not provided
        if not event_id and event_title:
            event_id = await _find_event_by_title(
                context, event_title, event_date_str, calendar_id, tz
            )
            if not event_id:
                return (
                    f"Could not find event matching '{event_title}'. "
                    f"Use list_events first to see available events with their event_id, "
                    f"then call delete_event with the exact event_id."
                )

        if not event_id:
            return "Error: Please provide either event_id or event_title to identify the event."

        try:
            # Get event details first for confirmation
            event = (
                context.calendar_service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
            event_title = event.get("summary", "Untitled")

            # Delete the event
            context.calendar_service.events().delete(
                calendarId=calendar_id, eventId=event_id
            ).execute()

            return f"Event deleted: **{event_title}**"

        except Exception as e:
            logger.error(f"Error deleting event: {e}", exc_info=True)
            return f"Error deleting event: {str(e)}"
