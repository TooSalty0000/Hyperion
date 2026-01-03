"""Utility functions for Polaris calendar tools."""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Calendar Cache - Local scratchpad for recent calendar fetches
# =============================================================================


@dataclass
class CachedEvent:
    """A cached calendar event with full metadata for easy lookup."""

    event_id: str
    summary: str  # Event title
    start_datetime: Optional[datetime] = None  # None for all-day events
    end_datetime: Optional[datetime] = None
    start_date: Optional[str] = None  # For all-day events: "2024-01-15"
    is_all_day: bool = False
    location: Optional[str] = None
    description: Optional[str] = None
    calendar_id: str = "primary"
    html_link: Optional[str] = None
    raw_event: Dict = field(default_factory=dict)  # Original API response

    @classmethod
    def from_api_event(cls, event: Dict, calendar_id: str = "primary") -> "CachedEvent":
        """Create a CachedEvent from a Google Calendar API response."""
        start = event.get("start", {})
        end = event.get("end", {})

        # Determine if all-day event
        is_all_day = "date" in start and "dateTime" not in start

        start_datetime = None
        end_datetime = None
        start_date = None

        if is_all_day:
            start_date = start.get("date")
        else:
            if "dateTime" in start:
                start_datetime = datetime.fromisoformat(
                    start["dateTime"].replace("Z", "+00:00")
                )
            if "dateTime" in end:
                end_datetime = datetime.fromisoformat(
                    end["dateTime"].replace("Z", "+00:00")
                )

        return cls(
            event_id=event.get("id", ""),
            summary=event.get("summary", "No title"),
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            start_date=start_date,
            is_all_day=is_all_day,
            location=event.get("location"),
            description=event.get("description"),
            calendar_id=calendar_id,
            html_link=event.get("htmlLink"),
            raw_event=event,
        )

    def matches_search(self, query: str) -> bool:
        """Check if this event matches a search query (case-insensitive)."""
        query_lower = query.lower()
        return (
            query_lower in self.summary.lower()
            or (self.description and query_lower in self.description.lower())
            or (self.location and query_lower in self.location.lower())
        )

    def get_date(self) -> Optional[datetime]:
        """Get the event's date (start_datetime or parsed start_date)."""
        if self.start_datetime:
            return self.start_datetime
        if self.start_date:
            return datetime.strptime(self.start_date, "%Y-%m-%d")
        return None

    def format_for_display(self) -> str:
        """Format event for Discord display with ID hint."""
        if self.is_all_day:
            time_str = "All day"
        elif self.start_datetime and self.end_datetime:
            time_str = f"{self.start_datetime.strftime('%I:%M %p')} - {self.end_datetime.strftime('%I:%M %p')}"
        elif self.start_datetime:
            time_str = self.start_datetime.strftime("%I:%M %p")
        else:
            time_str = "Unknown time"

        location_str = f" @ {self.location}" if self.location else ""
        id_hint = self.event_id[:8] if self.event_id else "?"

        return f"- **{self.summary}** ({time_str}){location_str} [id: {id_hint}...]"


class CalendarCache:
    """
    Local cache for recently fetched calendar events.

    This acts as a scratchpad that stores events from ListEventsTool,
    allowing UpdateEventTool and DeleteEventTool to look up events by
    name, date, or other attributes instead of requiring exact IDs.

    The cache is per-agent-instance and resets on restart.
    """

    def __init__(self, max_events: int = 100, ttl_minutes: int = 30):
        """
        Initialize the calendar cache.

        Args:
            max_events: Maximum number of events to cache
            ttl_minutes: How long to keep cached events (in minutes)
        """
        self._events: Dict[str, CachedEvent] = {}  # event_id -> CachedEvent
        self._max_events = max_events
        self._ttl_minutes = ttl_minutes
        self._last_fetch_time: Optional[datetime] = None
        self._last_fetch_range: Optional[tuple] = None  # (start_date, end_date)

    def cache_events(
        self,
        events: List[Dict],
        calendar_id: str = "primary",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Cache a list of events from Google Calendar API.

        Args:
            events: List of event dictionaries from API
            calendar_id: The calendar these events came from
            start_date: Start of the fetched date range
            end_date: End of the fetched date range

        Returns:
            Number of events cached
        """
        # Clear old events if cache is getting full
        if len(self._events) + len(events) > self._max_events:
            self._prune_oldest(len(events))

        cached_count = 0
        for event in events:
            event_id = event.get("id")
            if event_id:
                self._events[event_id] = CachedEvent.from_api_event(event, calendar_id)
                cached_count += 1

        self._last_fetch_time = datetime.now()
        self._last_fetch_range = (start_date, end_date)

        logger.info(
            f"[CalendarCache] Cached {cached_count} events "
            f"(total: {len(self._events)}, range: {start_date} to {end_date})"
        )
        return cached_count

    def _prune_oldest(self, space_needed: int):
        """Remove oldest events to make room for new ones."""
        if not self._events:
            return

        # Sort by event date, remove oldest
        sorted_events = sorted(
            self._events.values(),
            key=lambda e: e.get_date() or datetime.min,
        )

        to_remove = min(space_needed, len(sorted_events))
        for event in sorted_events[:to_remove]:
            del self._events[event.event_id]

        logger.debug(f"[CalendarCache] Pruned {to_remove} oldest events")

    def get_by_id(self, event_id: str) -> Optional[CachedEvent]:
        """Get a cached event by its exact ID."""
        return self._events.get(event_id)

    def get_by_partial_id(self, partial_id: str) -> Optional[CachedEvent]:
        """Get a cached event by a partial ID match."""
        partial_lower = partial_id.lower()
        for event_id, event in self._events.items():
            if event_id.lower().startswith(partial_lower):
                return event
        return None

    def search_by_title(
        self,
        title_query: str,
        date: Optional[datetime] = None,
        fuzzy: bool = True,
    ) -> List[CachedEvent]:
        """
        Search cached events by title.

        Args:
            title_query: Title or partial title to search for
            date: Optional date to filter by
            fuzzy: If True, match partial titles; if False, require exact match

        Returns:
            List of matching events, sorted by relevance
        """
        results = []
        query_lower = title_query.lower()

        for event in self._events.values():
            # Date filter
            if date:
                event_date = event.get_date()
                if not event_date or event_date.date() != date.date():
                    continue

            # Title match
            title_lower = event.summary.lower()
            if fuzzy:
                if query_lower in title_lower:
                    results.append(event)
            else:
                if query_lower == title_lower:
                    results.append(event)

        # Sort by start time
        results.sort(key=lambda e: e.get_date() or datetime.max)
        return results

    def search_by_date(
        self,
        date: datetime,
        include_all_day: bool = True,
    ) -> List[CachedEvent]:
        """
        Get all cached events for a specific date.

        Args:
            date: The date to search for
            include_all_day: Whether to include all-day events

        Returns:
            List of events on that date, sorted by start time
        """
        results = []
        target_date = date.date()

        for event in self._events.values():
            if event.is_all_day and not include_all_day:
                continue

            event_date = event.get_date()
            if event_date and event_date.date() == target_date:
                results.append(event)

        results.sort(key=lambda e: e.get_date() or datetime.max)
        return results

    def search(self, query: str) -> List[CachedEvent]:
        """
        General search across all cached events.

        Searches title, description, and location.

        Args:
            query: Search query

        Returns:
            List of matching events
        """
        results = [e for e in self._events.values() if e.matches_search(query)]
        results.sort(key=lambda e: e.get_date() or datetime.max)
        return results

    def find_best_match(
        self,
        title: Optional[str] = None,
        date: Optional[datetime] = None,
        partial_id: Optional[str] = None,
    ) -> Optional[CachedEvent]:
        """
        Find the best matching event given various hints.

        Priority: exact ID > partial ID > title + date > title only

        Args:
            title: Event title or partial title
            date: Target date
            partial_id: Partial event ID

        Returns:
            Best matching event or None
        """
        # Try partial ID first
        if partial_id:
            event = self.get_by_partial_id(partial_id)
            if event:
                return event

        # Try title search
        if title:
            matches = self.search_by_title(title, date=date)
            if matches:
                return matches[0]

            # Try without date filter
            if date:
                matches = self.search_by_title(title, date=None)
                if matches:
                    return matches[0]

        # Try date only
        if date:
            matches = self.search_by_date(date)
            if len(matches) == 1:
                return matches[0]

        return None

    def get_all(self) -> List[CachedEvent]:
        """Get all cached events, sorted by date."""
        events = list(self._events.values())
        events.sort(key=lambda e: e.get_date() or datetime.max)
        return events

    def clear(self):
        """Clear all cached events."""
        self._events.clear()
        self._last_fetch_time = None
        self._last_fetch_range = None
        logger.info("[CalendarCache] Cache cleared")

    def get_cache_summary(self) -> str:
        """Get a summary of the cache state for debugging."""
        if not self._events:
            return "Cache is empty. Use list_events to populate."

        oldest = min(
            (e.get_date() for e in self._events.values() if e.get_date()),
            default=None,
        )
        newest = max(
            (e.get_date() for e in self._events.values() if e.get_date()),
            default=None,
        )

        return (
            f"Cache: {len(self._events)} events "
            f"({oldest.strftime('%Y-%m-%d') if oldest else '?'} to "
            f"{newest.strftime('%Y-%m-%d') if newest else '?'}), "
            f"last fetch: {self._last_fetch_time.strftime('%H:%M:%S') if self._last_fetch_time else 'never'}"
        )

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)


@dataclass
class PolarisToolContext:
    """
    Context passed to Polaris tool execution.

    Provides access to Google Calendar service and config.
    """

    calendar_service: Optional[Any] = None
    config: Optional[Any] = None
    conversation_manager: Optional[Any] = None
    memory_manager: Optional[Any] = None
    discord_bot: Optional[Any] = None

    # Calendar cache for event lookups
    calendar_cache: Optional[CalendarCache] = None

    # Current context state
    current_channel_id: Optional[int] = None
    current_guild_id: Optional[int] = None
    user_id: Optional[int] = None
    current_agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


def get_default_token_path() -> str:
    """Get the default token storage path (~/.hyperion/polaris/token.json)."""
    from pathlib import Path

    token_dir = Path.home() / ".hyperion" / "polaris"
    token_dir.mkdir(parents=True, exist_ok=True)
    return str(token_dir / "token.json")


def get_calendar_service(
    credentials_path: str,
    token_path: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    headless: bool = False,
):
    """
    Initialize and return Google Calendar API service.

    Args:
        credentials_path: Path to credentials.json from Google Cloud Console
        token_path: Path to store/load OAuth token (defaults to ~/.hyperion/polaris/token.json)
        scopes: API scopes (defaults to calendar read/write)
        headless: If True, use console-based OAuth instead of browser

    Returns:
        Google Calendar API service object
    """
    if scopes is None:
        scopes = ["https://www.googleapis.com/auth/calendar"]

    if token_path is None:
        token_path = get_default_token_path()

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Google API libraries not installed. Run: "
            "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from e

    creds = None

    # Load existing token if available
    if os.path.exists(token_path):
        logger.info(f"Loading existing OAuth token from {token_path}")
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired OAuth token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_path}\n"
                    f"Download it from Google Cloud Console and set GOOGLE_CREDENTIALS_PATH"
                )

            logger.info("No valid token found - starting OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)

            if headless:
                # Console-based OAuth for headless environments
                logger.info("Using console-based OAuth (headless mode)")
                creds = flow.run_console()
            else:
                # Browser-based OAuth
                try:
                    logger.info("Opening browser for OAuth authentication...")
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.warning(f"Browser OAuth failed: {e}")
                    logger.info("Falling back to console-based OAuth...")
                    creds = flow.run_console()

        # Save the token for future use
        logger.info(f"Saving OAuth token to {token_path}")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    # Build and return the service
    service = build("calendar", "v3", credentials=creds)
    logger.info("Google Calendar service initialized successfully")
    return service


def load_calendar_service(
    token_path: Optional[str] = None,
    scopes: Optional[List[str]] = None,
):
    """
    Load Google Calendar service from existing token (no OAuth trigger).

    Args:
        token_path: Path to the OAuth token file
        scopes: API scopes (defaults to calendar read/write)

    Returns:
        Google Calendar API service object

    Raises:
        FileNotFoundError: If token file doesn't exist
        ValueError: If token is invalid or expired without refresh token
    """
    if scopes is None:
        scopes = ["https://www.googleapis.com/auth/calendar"]

    if token_path is None:
        token_path = get_default_token_path()

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Google API libraries not installed. Run: "
            "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from e

    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token file not found: {token_path}")

    logger.info(f"Loading OAuth token from {token_path}")
    creds = Credentials.from_authorized_user_file(token_path, scopes)

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        logger.info("Refreshing expired OAuth token...")
        creds.refresh(Request())
        # Save refreshed token
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    elif not creds.valid:
        raise ValueError("Token is invalid and cannot be refreshed")

    # Build and return the service
    service = build("calendar", "v3", credentials=creds)
    logger.info("Google Calendar service loaded successfully")
    return service


def start_oauth_flow(
    credentials_path: str,
    scopes: Optional[List[str]] = None,
) -> tuple:
    """
    Start OAuth flow and return the authorization URL (without opening browser).

    Args:
        credentials_path: Path to credentials.json from Google Cloud Console
        scopes: API scopes (defaults to calendar read/write)

    Returns:
        Tuple of (flow, authorization_url)
    """
    if scopes is None:
        scopes = ["https://www.googleapis.com/auth/calendar"]

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise ImportError(
            "Google API libraries not installed. Run: "
            "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from e

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}\n"
            f"Download it from Google Cloud Console and set GOOGLE_CREDENTIALS_PATH"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_path,
        scopes,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob"  # Manual copy/paste flow
    )

    auth_url, _ = flow.authorization_url(prompt="consent")

    logger.info(f"Generated OAuth authorization URL")
    return flow, auth_url


def complete_oauth_flow(
    flow,
    code: str,
    token_path: Optional[str] = None,
):
    """
    Complete OAuth flow by exchanging the authorization code for tokens.

    Args:
        flow: The OAuth flow object from start_oauth_flow
        code: The authorization code from Google
        token_path: Path to store the OAuth token

    Returns:
        Google Calendar API service object
    """
    if token_path is None:
        token_path = get_default_token_path()

    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Google API libraries not installed. Run: "
            "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from e

    # Exchange code for tokens
    logger.info("Exchanging authorization code for tokens...")
    flow.fetch_token(code=code)
    creds = flow.credentials

    # Save the token for future use
    logger.info(f"Saving OAuth token to {token_path}")
    token_dir = os.path.dirname(token_path)
    if token_dir:  # Only create directory if path has a directory component
        os.makedirs(token_dir, exist_ok=True)
    with open(token_path, "w") as token:
        token.write(creds.to_json())

    # Build and return the service
    service = build("calendar", "v3", credentials=creds)
    logger.info("Google Calendar service initialized successfully")
    return service


def check_calendar_auth_status(
    credentials_path: Optional[str] = None,
    token_path: Optional[str] = None,
) -> dict:
    """
    Check the current calendar authentication status without triggering OAuth.

    Returns:
        Dict with status info: {
            'authenticated': bool,
            'token_exists': bool,
            'token_path': str,
            'credentials_exists': bool,
            'credentials_path': str,
            'message': str
        }
    """
    if token_path is None:
        token_path = get_default_token_path()

    result = {
        'authenticated': False,
        'token_exists': os.path.exists(token_path),
        'token_path': token_path,
        'credentials_exists': credentials_path and os.path.exists(credentials_path),
        'credentials_path': credentials_path,
        'message': '',
    }

    if not result['credentials_exists']:
        result['message'] = "No Google credentials configured. Set GOOGLE_CREDENTIALS_PATH."
        return result

    if not result['token_exists']:
        result['message'] = "OAuth not completed. Run `!pauth` to authenticate."
        return result

    # Check if token is valid
    try:
        from google.oauth2.credentials import Credentials
        scopes = ["https://www.googleapis.com/auth/calendar"]
        creds = Credentials.from_authorized_user_file(token_path, scopes)

        if creds.valid:
            result['authenticated'] = True
            result['message'] = "Calendar authenticated and ready."
        elif creds.expired and creds.refresh_token:
            result['authenticated'] = True  # Can be refreshed
            result['message'] = "Token expired but can be refreshed."
        else:
            result['message'] = "Token invalid. Run `!pauth` to re-authenticate."
    except Exception as e:
        result['message'] = f"Token check failed: {e}"

    return result


def parse_datetime(
    date_str: str,
    time_str: Optional[str] = None,
    timezone: str = "UTC",
) -> datetime:
    """
    Parse date and optional time strings into a datetime object.

    Supports various formats:
    - "2024-01-15" (date only)
    - "2024-01-15 14:30" (date and time)
    - "tomorrow", "next monday", etc. (relative dates)
    - "2pm", "14:30", "2:30 PM" (time only - uses today)

    Args:
        date_str: Date string or relative date
        time_str: Optional time string
        timezone: Timezone name (default UTC)

    Returns:
        datetime object
    """
    import re
    from datetime import date

    now = datetime.now()

    # Handle relative dates
    date_lower = date_str.lower().strip()

    if date_lower == "today":
        result_date = now.date()
    elif date_lower == "tomorrow":
        result_date = (now + timedelta(days=1)).date()
    elif date_lower == "yesterday":
        result_date = (now - timedelta(days=1)).date()
    elif "next" in date_lower:
        # "next monday", "next week", etc.
        day_names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for i, day in enumerate(day_names):
            if day in date_lower:
                current_weekday = now.weekday()
                days_ahead = i - current_weekday
                if days_ahead <= 0:
                    days_ahead += 7
                result_date = (now + timedelta(days=days_ahead)).date()
                break
        else:
            if "week" in date_lower:
                result_date = (now + timedelta(days=7)).date()
            else:
                # Try to parse as regular date
                result_date = _parse_date_string(date_str)
    else:
        # Try to parse as regular date
        result_date = _parse_date_string(date_str)

    # Parse time if provided
    if time_str:
        result_time = _parse_time_string(time_str)
    else:
        # Check if time is included in date_str
        time_match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", date_lower, re.I)
        if time_match:
            result_time = _parse_time_string(time_match.group())
        else:
            result_time = datetime.min.time()  # Default to midnight

    return datetime.combine(result_date, result_time)


def _parse_date_string(date_str: str) -> "date":
    """Parse a date string into a date object."""
    from datetime import date

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d",
        "%b %d",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str.strip(), fmt)
            # If year not in format, use current year
            if "%Y" not in fmt:
                parsed = parsed.replace(year=datetime.now().year)
            return parsed.date()
        except ValueError:
            continue

    raise ValueError(f"Could not parse date: {date_str}")


def _parse_time_string(time_str: str) -> "datetime.time":
    """Parse a time string into a time object."""
    import re

    time_str = time_str.strip().lower()

    # Match patterns like "2pm", "2:30pm", "14:30", "2:30 PM"
    match = re.match(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", time_str, re.I)
    if not match:
        raise ValueError(f"Could not parse time: {time_str}")

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3)

    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0

    return datetime.min.replace(hour=hour, minute=minute).time()


def format_event_for_display(event: Dict[str, Any]) -> str:
    """
    Format a Google Calendar event for Discord display.

    Args:
        event: Event dictionary from Google Calendar API

    Returns:
        Formatted string for display
    """
    summary = event.get("summary", "No title")

    # Get start time
    start = event.get("start", {})
    if "dateTime" in start:
        start_dt = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
        start_str = start_dt.strftime("%I:%M %p")
    else:
        start_str = "All day"

    # Get end time
    end = event.get("end", {})
    if "dateTime" in end:
        end_dt = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00"))
        end_str = end_dt.strftime("%I:%M %p")
    else:
        end_str = ""

    # Build display string
    if end_str and start_str != "All day":
        time_range = f"{start_str} - {end_str}"
    else:
        time_range = start_str

    location = event.get("location", "")
    location_str = f" @ {location}" if location else ""

    return f"- **{summary}** ({time_range}){location_str}"


def format_datetime_for_api(dt: datetime, timezone: str = "UTC") -> str:
    """
    Format a datetime for Google Calendar API.

    Args:
        dt: datetime object
        timezone: Timezone name

    Returns:
        ISO format string with timezone
    """
    # For simplicity, format as UTC
    # In production, you'd want proper timezone handling
    return dt.isoformat() + "Z" if dt.tzinfo is None else dt.isoformat()
