"""Analysis and waiting tools for Canopus browser automation."""

import logging
from typing import List

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class WaitForElementTool(Tool):
    """Wait for an element to appear on the page."""

    @property
    def name(self) -> str:
        return "wait_for_element"

    @property
    def description(self) -> str:
        return (
            "Wait for an element to appear on the page. "
            "Useful for dynamic pages that load content asynchronously. "
            "States: 'visible' (default), 'attached', 'hidden', 'detached'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the element to wait for",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Maximum time to wait in milliseconds (default: 10000)",
                required=False,
            ),
            ToolParameter(
                name="state",
                type="string",
                description="State to wait for: 'visible', 'attached', 'hidden', 'detached' (default: 'visible')",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: str,
        timeout: int = 10000,
        state: str = "visible",
    ) -> str:
        """Wait for element."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        valid_states = ["visible", "attached", "hidden", "detached"]
        if state not in valid_states:
            return f"Error: Invalid state. Use one of: {', '.join(valid_states)}"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.wait_for_selector(selector, timeout=timeout, state=state)

            if result.get("success"):
                return f"Element found: {selector} (state: {state})"
            else:
                return f"Wait failed: {result.get('error', 'Element not found within timeout')}"

        except Exception as e:
            logger.error(f"Wait for element error: {e}", exc_info=True)
            return f"Wait for element error: {str(e)}"


class FindElementsTool(Tool):
    """Find and analyze elements matching a selector."""

    @property
    def name(self) -> str:
        return "find_elements"

    @property
    def description(self) -> str:
        return (
            "Find all elements matching a CSS selector and return information about them. "
            "Returns tag, id, class, text, href, type, name, value, placeholder, and visibility. "
            "Useful for discovering page structure, finding interactive elements, or debugging selectors."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector to find elements (e.g., 'button', '.nav-link', 'input[type=\"text\"]')",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of elements to return (default: 20)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: str,
        limit: int = 20,
    ) -> str:
        """Find elements."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.find_elements(selector, limit=limit)

            if result.get("success"):
                elements = result.get("elements", [])
                if not elements:
                    return f"No elements found matching: {selector}"

                lines = [f"Found {result['count']} elements matching '{selector}':"]

                for el in elements:
                    # Build element description
                    parts = [f"[{el['index']}] <{el['tag']}>"]

                    if el.get("id"):
                        parts.append(f"id={el['id']}")
                    if el.get("className"):
                        cls = el["className"][:30]
                        parts.append(f"class={cls}")
                    if el.get("type"):
                        parts.append(f"type={el['type']}")
                    if el.get("name"):
                        parts.append(f"name={el['name']}")
                    if el.get("href"):
                        href = el["href"][:40]
                        parts.append(f"href={href}...")
                    if not el.get("isVisible"):
                        parts.append("(hidden)")

                    lines.append("  " + " ".join(parts))

                    # Add text/value on separate line if present
                    if el.get("text"):
                        text = el["text"][:60]
                        lines.append(f"      text: \"{text}\"")
                    if el.get("value"):
                        lines.append(f"      value: \"{el['value']}\"")
                    if el.get("placeholder"):
                        lines.append(f"      placeholder: \"{el['placeholder']}\"")

                return "\n".join(lines)
            else:
                return f"Find elements failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Find elements error: {e}", exc_info=True)
            return f"Find elements error: {str(e)}"
