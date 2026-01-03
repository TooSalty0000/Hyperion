"""Interaction tools for Canopus browser automation."""

import logging
from typing import List

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class ClickTool(Tool):
    """Click an element on the page."""

    @property
    def name(self) -> str:
        return "click"

    @property
    def description(self) -> str:
        return (
            "Click an element using a CSS selector from the snapshot. "
            "Use the selector= value from get_snapshot output. "
            "Examples: '#submit-btn', '[data-testid=\"login\"]', 'text=\"Submit\"'. "
            "If no selector is available, use click_by_index with the [N] index instead."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector from snapshot (e.g., '#id', 'text=\"Button\"')",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, selector: str) -> str:
        """Click element."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.click(selector)

            if result.get("success"):
                return f"Clicked element: {selector}"
            else:
                return f"Click failed: {result.get('error', 'Unknown error')}. Try click_by_index instead."

        except Exception as e:
            logger.error(f"Click error: {e}", exc_info=True)
            return f"Click error: {str(e)}. Try click_by_index instead."


class ClickByIndexTool(Tool):
    """Click an element by its index from the snapshot."""

    @property
    def name(self) -> str:
        return "click_by_index"

    @property
    def description(self) -> str:
        return (
            "Click an interactive element by its index [N] from the get_snapshot output. "
            "Use this when the element has no selector= value or when click() fails. "
            "The index is the number in brackets, e.g., [0], [5], [12]."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="index",
                type="integer",
                description="Element index from snapshot (the number in [N])",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, index: int) -> str:
        """Click element by index."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.click_by_index(index)

            if result.get("success"):
                tag = result.get("tag", "element")
                text = result.get("text", "")
                if text:
                    return f"Clicked [{index}] {tag}: \"{text}\""
                return f"Clicked [{index}] {tag}"
            else:
                return f"Click by index failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Click by index error: {e}", exc_info=True)
            return f"Click by index error: {str(e)}"


class TypeTextTool(Tool):
    """Type text into an input element."""

    @property
    def name(self) -> str:
        return "type_text"

    @property
    def description(self) -> str:
        return (
            "Type text into an input element on the page. "
            "Use clear=True to replace existing text, or False to append. "
            "Common selectors: 'input[name=\"...\"]', '#search', '.input-field', "
            "'[placeholder=\"...\"]'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the input element",
                required=True,
            ),
            ToolParameter(
                name="text",
                type="string",
                description="The text to type into the element",
                required=True,
            ),
            ToolParameter(
                name="clear",
                type="boolean",
                description="Clear existing text before typing (default: True)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: str,
        text: str,
        clear: bool = True,
    ) -> str:
        """Type text into element."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.type_text(selector, text, clear=clear)

            if result.get("success"):
                preview = text[:50] + "..." if len(text) > 50 else text
                return f"Typed into {selector}: '{preview}'"
            else:
                return f"Type failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Type error: {e}", exc_info=True)
            return f"Type error: {str(e)}"


class PressKeyTool(Tool):
    """Press a keyboard key."""

    @property
    def name(self) -> str:
        return "press_key"

    @property
    def description(self) -> str:
        return (
            "Press a keyboard key. "
            "Common keys: 'Enter', 'Tab', 'Escape', 'Backspace', 'Delete', "
            "'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', "
            "'Control+a', 'Control+c', 'Control+v'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="key",
                type="string",
                description="The key to press (e.g., 'Enter', 'Tab', 'Escape')",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, key: str) -> str:
        """Press key."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.press_key(key)

            if result.get("success"):
                return f"Pressed key: {key}"
            else:
                return f"Key press failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Key press error: {e}", exc_info=True)
            return f"Key press error: {str(e)}"


class ScrollTool(Tool):
    """Scroll the page."""

    @property
    def name(self) -> str:
        return "scroll"

    @property
    def description(self) -> str:
        return (
            "Scroll the page up or down. "
            "Use this to reveal more content on long pages. "
            "The amount parameter controls how many pixels to scroll."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="direction",
                type="string",
                description="Direction to scroll: 'up' or 'down'",
                required=True,
            ),
            ToolParameter(
                name="amount",
                type="integer",
                description="Pixels to scroll (default: 500)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        direction: str,
        amount: int = 500,
    ) -> str:
        """Scroll page."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        if direction not in ["up", "down"]:
            return "Error: Direction must be 'up' or 'down'"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.scroll(direction=direction, amount=amount)

            if result.get("success"):
                return f"Scrolled {direction} {amount}px"
            else:
                return f"Scroll failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Scroll error: {e}", exc_info=True)
            return f"Scroll error: {str(e)}"


class HoverTool(Tool):
    """Hover over an element on the page."""

    @property
    def name(self) -> str:
        return "hover"

    @property
    def description(self) -> str:
        return (
            "Hover the mouse over an element on the page. "
            "Useful for triggering dropdown menus, tooltips, or hover states. "
            "Use CSS selectors like '#menu-item', '.nav-link', or 'button'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the element to hover over",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, selector: str) -> str:
        """Hover over element."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.hover(selector)

            if result.get("success"):
                return f"Hovering over element: {selector}"
            else:
                return f"Hover failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Hover error: {e}", exc_info=True)
            return f"Hover error: {str(e)}"


class DoubleClickTool(Tool):
    """Double-click an element on the page."""

    @property
    def name(self) -> str:
        return "double_click"

    @property
    def description(self) -> str:
        return (
            "Double-click an element on the page. "
            "Useful for text selection, opening items, or triggering double-click actions."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the element to double-click",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, selector: str) -> str:
        """Double-click element."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.double_click(selector)

            if result.get("success"):
                return f"Double-clicked element: {selector}"
            else:
                return f"Double-click failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Double-click error: {e}", exc_info=True)
            return f"Double-click error: {str(e)}"


class SelectOptionTool(Tool):
    """Select an option from a dropdown."""

    @property
    def name(self) -> str:
        return "select_option"

    @property
    def description(self) -> str:
        return (
            "Select an option from a dropdown/select element. "
            "The value should match the option's value attribute or visible text."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the select element",
                required=True,
            ),
            ToolParameter(
                name="value",
                type="string",
                description="Value or text of the option to select",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, selector: str, value: str) -> str:
        """Select option from dropdown."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.select_option(selector, value)

            if result.get("success"):
                return f"Selected '{value}' in {selector}"
            else:
                return f"Select option failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Select option error: {e}", exc_info=True)
            return f"Select option error: {str(e)}"
