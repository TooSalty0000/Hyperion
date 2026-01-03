"""Form handling tools for Canopus browser automation."""

import logging
from typing import List, Dict, Any

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class FillFormTool(Tool):
    """Fill multiple form fields at once."""

    @property
    def name(self) -> str:
        return "fill_form"

    @property
    def description(self) -> str:
        return (
            "Fill multiple form fields at once. "
            "Provide a dictionary mapping selectors to values. "
            "Example: {'#name': 'John', '#email': 'john@example.com', '#age': '30'}"
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="fields",
                type="object",
                description="Dictionary mapping CSS selectors to values (e.g., {'#name': 'John', '#email': 'test@example.com'})",
                required=True,
            ),
            ToolParameter(
                name="submit_selector",
                type="string",
                description="Optional selector for submit button to click after filling",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        fields: Dict[str, Any],
        submit_selector: str = None,
    ) -> str:
        """Fill form fields."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        if not fields:
            return "Error: No fields provided to fill."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            filled = []
            errors = []

            for selector, value in fields.items():
                result = await session.type_text(selector, str(value), clear=True)
                if result.get("success"):
                    filled.append(selector)
                else:
                    errors.append(f"{selector}: {result.get('error', 'Unknown error')}")

            # Submit if requested
            submitted = False
            if submit_selector and filled:
                click_result = await session.click(submit_selector)
                if click_result.get("success"):
                    submitted = True
                else:
                    errors.append(f"Submit failed: {click_result.get('error')}")

            # Build result message
            result_parts = []
            if filled:
                result_parts.append(f"Filled {len(filled)} field(s): {', '.join(filled)}")
            if submitted:
                result_parts.append(f"Submitted form via {submit_selector}")
            if errors:
                result_parts.append(f"Errors: {'; '.join(errors)}")

            return "\n".join(result_parts) if result_parts else "No actions performed."

        except Exception as e:
            logger.error(f"Fill form error: {e}", exc_info=True)
            return f"Fill form error: {str(e)}"
