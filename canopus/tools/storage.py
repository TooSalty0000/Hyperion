"""Cookie and storage management tools for Canopus browser automation."""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class GetCookiesTool(Tool):
    """Get cookies from the browser."""

    @property
    def name(self) -> str:
        return "get_cookies"

    @property
    def description(self) -> str:
        return (
            "Get all cookies from the browser context. "
            "Optionally filter by specific URLs. "
            "Useful for debugging authentication or session state."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="urls",
                type="array",
                description="Optional list of URLs to filter cookies (e.g., ['https://example.com'])",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        urls: Optional[List[str]] = None,
    ) -> str:
        """Get cookies."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.get_cookies(urls)

            if result.get("success"):
                cookies = result.get("cookies", [])
                if not cookies:
                    return "No cookies found."

                lines = [f"Found {result['count']} cookies:"]
                for c in cookies[:20]:  # Limit display
                    flags = []
                    if c.get("httpOnly"):
                        flags.append("httpOnly")
                    if c.get("secure"):
                        flags.append("secure")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    lines.append(f"  {c['name']}: {c['value']} ({c['domain']}){flag_str}")

                if len(cookies) > 20:
                    lines.append(f"  ... and {len(cookies) - 20} more")

                return "\n".join(lines)
            else:
                return f"Get cookies failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Get cookies error: {e}", exc_info=True)
            return f"Get cookies error: {str(e)}"


class SetCookieTool(Tool):
    """Set a cookie in the browser."""

    @property
    def name(self) -> str:
        return "set_cookie"

    @property
    def description(self) -> str:
        return (
            "Set a cookie in the browser context. "
            "Domain defaults to current page domain if not specified."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="Cookie name",
                required=True,
            ),
            ToolParameter(
                name="value",
                type="string",
                description="Cookie value",
                required=True,
            ),
            ToolParameter(
                name="domain",
                type="string",
                description="Cookie domain (optional, uses current page domain)",
                required=False,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Cookie path (default: '/')",
                required=False,
            ),
            ToolParameter(
                name="expires",
                type="integer",
                description="Expiration timestamp in seconds since epoch",
                required=False,
            ),
            ToolParameter(
                name="http_only",
                type="boolean",
                description="HTTP-only flag (default: false)",
                required=False,
            ),
            ToolParameter(
                name="secure",
                type="boolean",
                description="Secure flag (default: false)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        name: str,
        value: str,
        domain: Optional[str] = None,
        path: str = "/",
        expires: Optional[int] = None,
        http_only: bool = False,
        secure: bool = False,
    ) -> str:
        """Set cookie."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_cookie(
                name=name,
                value=value,
                domain=domain,
                path=path,
                expires=expires,
                http_only=http_only,
                secure=secure,
            )

            if result.get("success"):
                return f"Cookie '{name}' set for domain: {result.get('domain')}"
            else:
                return f"Set cookie failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set cookie error: {e}", exc_info=True)
            return f"Set cookie error: {str(e)}"


class ClearCookiesTool(Tool):
    """Clear all cookies from the browser."""

    @property
    def name(self) -> str:
        return "clear_cookies"

    @property
    def description(self) -> str:
        return "Clear all cookies from the browser context."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Clear cookies."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.clear_cookies()

            if result.get("success"):
                return "All cookies cleared."
            else:
                return f"Clear cookies failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Clear cookies error: {e}", exc_info=True)
            return f"Clear cookies error: {str(e)}"


class GetLocalStorageTool(Tool):
    """Get localStorage items from the page."""

    @property
    def name(self) -> str:
        return "get_local_storage"

    @property
    def description(self) -> str:
        return (
            "Get all localStorage items from the current page. "
            "Useful for debugging application state."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Get localStorage."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.get_local_storage()

            if result.get("success"):
                storage = result.get("storage", {})
                if not storage:
                    return "localStorage is empty."

                lines = [f"localStorage ({result['count']} items):"]
                for key, value in list(storage.items())[:20]:
                    val_preview = value[:50] + "..." if len(value) > 50 else value
                    lines.append(f"  {key}: {val_preview}")

                if len(storage) > 20:
                    lines.append(f"  ... and {len(storage) - 20} more")

                return "\n".join(lines)
            else:
                return f"Get localStorage failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Get localStorage error: {e}", exc_info=True)
            return f"Get localStorage error: {str(e)}"


class SetLocalStorageTool(Tool):
    """Set a localStorage item on the page."""

    @property
    def name(self) -> str:
        return "set_local_storage"

    @property
    def description(self) -> str:
        return "Set a localStorage item on the current page."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="key",
                type="string",
                description="Storage key",
                required=True,
            ),
            ToolParameter(
                name="value",
                type="string",
                description="Storage value",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        key: str,
        value: str,
    ) -> str:
        """Set localStorage item."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_local_storage(key, value)

            if result.get("success"):
                return f"localStorage['{key}'] set successfully."
            else:
                return f"Set localStorage failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set localStorage error: {e}", exc_info=True)
            return f"Set localStorage error: {str(e)}"


class ClearStorageTool(Tool):
    """Clear browser storage."""

    @property
    def name(self) -> str:
        return "clear_storage"

    @property
    def description(self) -> str:
        return (
            "Clear browser storage (localStorage, sessionStorage, or both). "
            "Use 'local', 'session', or 'all' for storage_type."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="storage_type",
                type="string",
                description="Type of storage to clear: 'local', 'session', or 'all' (default: 'all')",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        storage_type: str = "all",
    ) -> str:
        """Clear storage."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        if storage_type not in ["local", "session", "all"]:
            return "Error: storage_type must be 'local', 'session', or 'all'"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.clear_storage(storage_type)

            if result.get("success"):
                return f"Cleared {result['cleared']} storage."
            else:
                return f"Clear storage failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Clear storage error: {e}", exc_info=True)
            return f"Clear storage error: {str(e)}"
