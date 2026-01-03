"""Network interception and monitoring tools for Canopus browser automation."""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class EnableNetworkMonitoringTool(Tool):
    """Enable network request/response monitoring."""

    @property
    def name(self) -> str:
        return "enable_network_monitoring"

    @property
    def description(self) -> str:
        return (
            "Enable network request and response monitoring. "
            "After enabling, all requests and responses will be logged. "
            "Use get_network_log to retrieve the captured data."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Enable network monitoring."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.enable_request_interception()

            if result.get("success"):
                return "Network monitoring enabled. All requests/responses will be logged."
            else:
                return f"Enable monitoring failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Enable network monitoring error: {e}", exc_info=True)
            return f"Enable network monitoring error: {str(e)}"


class GetNetworkLogTool(Tool):
    """Get captured network requests and responses."""

    @property
    def name(self) -> str:
        return "get_network_log"

    @property
    def description(self) -> str:
        return (
            "Get captured network requests and responses. "
            "Enable network monitoring first with enable_network_monitoring. "
            "Filter by 'request' or 'response' type."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filter_type",
                type="string",
                description="Filter by type: 'request', 'response', or None for all",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum entries to return (default: 50)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        filter_type: Optional[str] = None,
        limit: int = 50,
    ) -> str:
        """Get network log."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.get_network_log(filter_type, limit)

            if result.get("success"):
                log = result.get("log", [])
                if not log:
                    if result.get("message"):
                        return result["message"]
                    return "No network activity logged."

                lines = [f"Network Log ({result['count']}/{result['total']} entries):"]

                for entry in log[:30]:  # Limit display
                    entry_type = entry.get("type", "unknown")
                    url = entry.get("url", "")
                    if len(url) > 60:
                        url = url[:60] + "..."

                    if entry_type == "request":
                        method = entry.get("method", "?")
                        resource = entry.get("resource_type", "")
                        lines.append(f"  → {method} {url} [{resource}]")
                    else:
                        status = entry.get("status", "?")
                        lines.append(f"  ← {status} {url}")

                if len(log) > 30:
                    lines.append(f"  ... and {len(log) - 30} more entries")

                return "\n".join(lines)
            else:
                return f"Get network log failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Get network log error: {e}", exc_info=True)
            return f"Get network log error: {str(e)}"


class ClearNetworkLogTool(Tool):
    """Clear the captured network log."""

    @property
    def name(self) -> str:
        return "clear_network_log"

    @property
    def description(self) -> str:
        return "Clear the captured network request/response log."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Clear network log."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.clear_network_log()

            if result.get("success"):
                return "Network log cleared."
            else:
                return f"Clear network log failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Clear network log error: {e}", exc_info=True)
            return f"Clear network log error: {str(e)}"


class BlockResourcesTool(Tool):
    """Block specific resource types from loading."""

    @property
    def name(self) -> str:
        return "block_resources"

    @property
    def description(self) -> str:
        return (
            "Block specific resource types from loading. "
            "Useful for faster page loads or testing without images/scripts. "
            "Types: image, stylesheet, script, font, xhr, fetch, websocket, media"
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="resource_types",
                type="array",
                description="List of resource types to block (e.g., ['image', 'stylesheet'])",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        resource_types: List[str],
    ) -> str:
        """Block resources."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        valid_types = [
            "image", "stylesheet", "script", "font", "xhr",
            "fetch", "websocket", "media", "document", "other"
        ]

        invalid = [t for t in resource_types if t not in valid_types]
        if invalid:
            return f"Error: Invalid resource types: {invalid}. Valid: {valid_types}"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.block_resources(resource_types)

            if result.get("success"):
                return f"Blocking resources: {', '.join(result['blocked_types'])}"
            else:
                return f"Block resources failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Block resources error: {e}", exc_info=True)
            return f"Block resources error: {str(e)}"
