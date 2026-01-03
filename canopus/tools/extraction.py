"""Extraction tools for Canopus browser automation."""

import logging
import io
from datetime import datetime
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class ScreenshotTool(Tool):
    """Take a screenshot of the page or element."""

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return (
            "Take a screenshot of the current page or a specific element. "
            "Screenshots are saved and can be shared in Discord. "
            "Use full_page=True to capture the entire scrollable page. "
            "Use selector to capture a specific element."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filename",
                type="string",
                description="Filename for the screenshot (optional, auto-generated if not provided)",
                required=False,
            ),
            ToolParameter(
                name="full_page",
                type="boolean",
                description="Capture full scrollable page instead of viewport (default: False)",
                required=False,
            ),
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector to capture specific element instead of page",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        filename: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> str:
        """Take screenshot."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"

            # Get full path
            path = context.browser_manager.get_screenshot_path(filename)

            result = await session.screenshot(
                path=str(path),
                full_page=full_page,
                selector=selector,
            )

            if result.get("success"):
                # If Discord bot available, send the screenshot
                if context.discord_bot and context.current_channel_id:
                    try:
                        import discord
                        channel = context.discord_bot.get_channel(context.current_channel_id)
                        if channel:
                            file = discord.File(str(path), filename=filename)
                            await channel.send(
                                f"📸 Screenshot: {session.current_url}",
                                file=file,
                            )
                            return f"Screenshot taken and sent to Discord: {filename}"
                    except Exception as e:
                        logger.warning(f"Failed to send screenshot to Discord: {e}")

                return f"Screenshot saved: {path}"
            else:
                return f"Screenshot failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Screenshot error: {e}", exc_info=True)
            return f"Screenshot error: {str(e)}"


class ExtractTextTool(Tool):
    """Extract text content from the page."""

    @property
    def name(self) -> str:
        return "extract_text"

    @property
    def description(self) -> str:
        return (
            "Extract text content from the page or a specific element. "
            "Use selector to target a specific element, or leave empty for full page text. "
            "Useful for getting article content, form values, or specific data."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for element to extract text from (optional, full page if not provided)",
                required=False,
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum characters to return (default: 2000)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: Optional[str] = None,
        max_length: int = 2000,
    ) -> str:
        """Extract text from page."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            if selector:
                # Extract from specific element
                js_code = f'document.querySelector("{selector}")?.innerText || ""'
            else:
                # Extract from body
                js_code = 'document.body.innerText'

            result = await session.evaluate(js_code)

            if result.get("success"):
                text = result.get("result", "")
                if len(text) > max_length:
                    text = text[:max_length] + "\n... [truncated]"

                target = f"element '{selector}'" if selector else "page"
                return f"Text from {target}:\n{text}"
            else:
                return f"Extraction failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Extract text error: {e}", exc_info=True)
            return f"Extract text error: {str(e)}"


class EvaluateJsTool(Tool):
    """Evaluate JavaScript on the page."""

    @property
    def name(self) -> str:
        return "evaluate_js"

    @property
    def description(self) -> str:
        return (
            "Evaluate JavaScript expression on the page. "
            "Use for custom data extraction, DOM manipulation, or accessing page APIs. "
            "The expression should return a value (string, number, object, etc.)."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="JavaScript expression to evaluate (e.g., 'document.title', 'window.location.href')",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, expression: str) -> str:
        """Evaluate JavaScript."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.evaluate(expression)

            if result.get("success"):
                value = result.get("result")
                # Format result for display
                if isinstance(value, (dict, list)):
                    import json
                    formatted = json.dumps(value, indent=2, default=str)
                    if len(formatted) > 2000:
                        formatted = formatted[:2000] + "\n... [truncated]"
                    return f"JavaScript result:\n```json\n{formatted}\n```"
                else:
                    return f"JavaScript result: {value}"
            else:
                return f"JavaScript evaluation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Evaluate JS error: {e}", exc_info=True)
            return f"Evaluate JS error: {str(e)}"


class GetLinksTool(Tool):
    """Extract all links from the page."""

    @property
    def name(self) -> str:
        return "get_links"

    @property
    def description(self) -> str:
        return (
            "Extract all links from the page or a specific container element. "
            "Returns a list of URLs with their link text. "
            "Useful for finding navigation options, article links, or any hyperlinks."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="Optional CSS selector to limit search to a container element",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: Optional[str] = None,
    ) -> str:
        """Extract links from page."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.get_links(selector)

            if result.get("success"):
                links = result.get("links", [])
                if not links:
                    return "No links found on the page."

                lines = [f"Found {result['count']} links:"]
                for i, link in enumerate(links[:50]):  # Limit to 50 links
                    text = link.get("text", "").strip()[:40] or "(no text)"
                    href = link.get("href", "")
                    if len(href) > 60:
                        href = href[:60] + "..."
                    lines.append(f"  [{i}] {text} → {href}")

                if len(links) > 50:
                    lines.append(f"  ... and {len(links) - 50} more links")

                return "\n".join(lines)
            else:
                return f"Get links failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Get links error: {e}", exc_info=True)
            return f"Get links error: {str(e)}"


class GetPageInfoTool(Tool):
    """Get comprehensive information about the current page."""

    @property
    def name(self) -> str:
        return "get_page_info"

    @property
    def description(self) -> str:
        return (
            "Get comprehensive information about the current page including: "
            "URL, title, meta description, viewport dimensions, scroll position, "
            "and counts of key elements (links, images, forms, inputs, buttons). "
            "Useful for understanding page structure before interacting."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """Get page info."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.get_page_info()

            if result.get("success"):
                viewport = result.get("viewport", {})
                counts = result.get("counts", {})

                lines = [
                    "Page Information:",
                    f"  URL: {result.get('url', 'N/A')}",
                    f"  Title: {result.get('title', 'N/A')}",
                ]

                if result.get("description"):
                    desc = result["description"][:100]
                    lines.append(f"  Description: {desc}...")

                if result.get("language"):
                    lines.append(f"  Language: {result['language']}")

                lines.extend([
                    "",
                    "Viewport:",
                    f"  Size: {viewport.get('width', 0)}x{viewport.get('height', 0)}",
                    f"  Scroll: ({viewport.get('scrollX', 0)}, {viewport.get('scrollY', 0)})",
                    f"  Page Size: {viewport.get('scrollWidth', 0)}x{viewport.get('scrollHeight', 0)}",
                    "",
                    "Element Counts:",
                    f"  Links: {counts.get('links', 0)}",
                    f"  Images: {counts.get('images', 0)}",
                    f"  Forms: {counts.get('forms', 0)}",
                    f"  Inputs: {counts.get('inputs', 0)}",
                    f"  Buttons: {counts.get('buttons', 0)}",
                ])

                return "\n".join(lines)
            else:
                return f"Get page info failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Get page info error: {e}", exc_info=True)
            return f"Get page info error: {str(e)}"
