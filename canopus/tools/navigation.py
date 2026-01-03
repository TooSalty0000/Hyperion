"""Navigation tools for Canopus browser automation."""

import logging
from typing import List

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class NavigateToTool(Tool):
    """Navigate the browser to a URL."""

    @property
    def name(self) -> str:
        return "navigate_to"

    @property
    def description(self) -> str:
        return (
            "Navigate the browser to a URL. Creates a browser session if one doesn't exist. "
            "Use this to visit websites, web apps, or any URL. "
            "After navigation, use get_snapshot to understand the page structure."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL to navigate to (e.g., 'https://example.com')",
                required=True,
            ),
            ToolParameter(
                name="wait_until",
                type="string",
                description="When to consider navigation complete: 'load' (default), 'domcontentloaded', or 'networkidle'",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        url: str,
        wait_until: str = "load",
    ) -> str:
        """Navigate to URL."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            # Get or create browser session
            session = await context.browser_manager.get_or_create(context.current_channel_id)

            # Navigate
            result = await session.navigate(url, wait_until=wait_until)

            if result.get("success"):
                return (
                    f"Navigated to: {result['url']}\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"Status: {result.get('status', 'N/A')}\n"
                    f"Use get_snapshot to see the page structure."
                )
            else:
                return f"Navigation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Navigation error: {e}", exc_info=True)
            return f"Navigation error: {str(e)}"


class GoBackTool(Tool):
    """Navigate back in browser history."""

    @property
    def name(self) -> str:
        return "go_back"

    @property
    def description(self) -> str:
        return (
            "Navigate back to the previous page in browser history. "
            "Use this to return to a page after clicking a link or navigating away."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Go back in history."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.go_back()

            if result.get("success"):
                return (
                    f"Navigated back to: {result['url']}\n"
                    f"Title: {result.get('title', 'N/A')}"
                )
            else:
                return f"Go back failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Go back error: {e}", exc_info=True)
            return f"Go back error: {str(e)}"


class GetSnapshotTool(Tool):
    """Get a snapshot of the current page's structure and content."""

    @property
    def name(self) -> str:
        return "get_snapshot"

    @property
    def description(self) -> str:
        return (
            "Get a snapshot of the current page. "
            "This provides a text representation including page headings, "
            "interactive elements (buttons, links, inputs with indices), and content preview. "
            "ALWAYS use this after navigating to understand what's on the page "
            "before taking actions like clicking or typing."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Get page snapshot."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            snapshot = await session.get_snapshot()

            # Truncate if too long
            content = snapshot.content
            if len(content) > 4000:
                content = content[:4000] + "\n... [truncated]"

            return (
                f"Page Snapshot:\n"
                f"URL: {snapshot.url}\n"
                f"Title: {snapshot.title}\n"
                f"---\n"
                f"{content}"
            )

        except Exception as e:
            logger.error(f"Snapshot error: {e}", exc_info=True)
            return f"Snapshot error: {str(e)}"


class RefreshTool(Tool):
    """Refresh the current page."""

    @property
    def name(self) -> str:
        return "refresh"

    @property
    def description(self) -> str:
        return (
            "Refresh/reload the current page. "
            "Use this to get the latest content or reset page state."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Refresh page."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.refresh()

            if result.get("success"):
                return (
                    f"Page refreshed.\n"
                    f"URL: {result['url']}\n"
                    f"Title: {result.get('title', 'N/A')}"
                )
            else:
                return f"Refresh failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Refresh error: {e}", exc_info=True)
            return f"Refresh error: {str(e)}"


class GoForwardTool(Tool):
    """Navigate forward in browser history."""

    @property
    def name(self) -> str:
        return "go_forward"

    @property
    def description(self) -> str:
        return (
            "Navigate forward in browser history. "
            "Use this after going back to return to a later page."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Go forward in history."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.go_forward()

            if result.get("success"):
                return (
                    f"Navigated forward to: {result['url']}\n"
                    f"Title: {result.get('title', 'N/A')}"
                )
            else:
                return f"Go forward failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Go forward error: {e}", exc_info=True)
            return f"Go forward error: {str(e)}"


class ListTabsTool(Tool):
    """List all open browser tabs."""

    @property
    def name(self) -> str:
        return "list_tabs"

    @property
    def description(self) -> str:
        return (
            "List all open browser tabs with their URLs and titles. "
            "Shows which tab is currently active."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """List tabs."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            tabs = await session.list_tabs()

            if not tabs:
                return "No tabs open."

            lines = [f"Open Tabs ({len(tabs)}):"]
            for tab in tabs:
                current = " (current)" if tab["is_current"] else ""
                url = tab["url"]
                if len(url) > 60:
                    url = url[:60] + "..."
                lines.append(f"  [{tab['index']}]{current} {tab['title'][:40]} - {url}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"List tabs error: {e}", exc_info=True)
            return f"List tabs error: {str(e)}"


class NewTabTool(Tool):
    """Open a new browser tab."""

    @property
    def name(self) -> str:
        return "new_tab"

    @property
    def description(self) -> str:
        return (
            "Open a new browser tab. Optionally navigate to a URL. "
            "The new tab becomes the active tab."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="Optional URL to navigate to in the new tab",
                required=False,
            ),
        ]

    async def execute(self, context: ToolContext, url: str = None) -> str:
        """Open new tab."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.new_tab(url)

            if result.get("success"):
                if url:
                    return (
                        f"Opened new tab and navigated to: {result['url']}\n"
                        f"Title: {result.get('title', 'N/A')}\n"
                        f"Total tabs: {result['tab_count']}"
                    )
                else:
                    return f"Opened new blank tab. Total tabs: {result['tab_count']}"
            else:
                return f"New tab failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"New tab error: {e}", exc_info=True)
            return f"New tab error: {str(e)}"


class SwitchTabTool(Tool):
    """Switch to a different browser tab."""

    @property
    def name(self) -> str:
        return "switch_tab"

    @property
    def description(self) -> str:
        return (
            "Switch to a different browser tab by index. "
            "Use list_tabs first to see available tabs and their indices."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="index",
                type="integer",
                description="The tab index to switch to (0-based)",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, index: int) -> str:
        """Switch tab."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.switch_tab(index)

            if result.get("success"):
                return (
                    f"Switched to tab {index}.\n"
                    f"URL: {result['url']}\n"
                    f"Title: {result.get('title', 'N/A')}"
                )
            else:
                return f"Switch tab failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Switch tab error: {e}", exc_info=True)
            return f"Switch tab error: {str(e)}"


class CloseTabTool(Tool):
    """Close a browser tab."""

    @property
    def name(self) -> str:
        return "close_tab"

    @property
    def description(self) -> str:
        return (
            "Close a browser tab by index. If no index provided, closes current tab. "
            "Cannot close the last remaining tab."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="index",
                type="integer",
                description="Tab index to close (optional, closes current if not specified)",
                required=False,
            ),
        ]

    async def execute(self, context: ToolContext, index: int = None) -> str:
        """Close tab."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.close_tab(index)

            if result.get("success"):
                return (
                    f"Closed tab {result['closed_index']}. "
                    f"Remaining tabs: {result['remaining_tabs']}"
                )
            else:
                return f"Close tab failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Close tab error: {e}", exc_info=True)
            return f"Close tab error: {str(e)}"
