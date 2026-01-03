"""Device emulation and viewport tools for Canopus browser automation."""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class SetViewportTool(Tool):
    """Set the browser viewport size."""

    @property
    def name(self) -> str:
        return "set_viewport"

    @property
    def description(self) -> str:
        return (
            "Set the browser viewport size. "
            "Useful for testing responsive designs at different screen sizes."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="width",
                type="integer",
                description="Viewport width in pixels",
                required=True,
            ),
            ToolParameter(
                name="height",
                type="integer",
                description="Viewport height in pixels",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        width: int,
        height: int,
    ) -> str:
        """Set viewport."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        if width < 200 or width > 5000:
            return "Error: Width must be between 200 and 5000 pixels"
        if height < 200 or height > 5000:
            return "Error: Height must be between 200 and 5000 pixels"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_viewport(width, height)

            if result.get("success"):
                viewport = result["viewport"]
                return f"Viewport set to {viewport['width']}x{viewport['height']}"
            else:
                return f"Set viewport failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set viewport error: {e}", exc_info=True)
            return f"Set viewport error: {str(e)}"


class EmulateDeviceTool(Tool):
    """Emulate a specific mobile device or screen size."""

    @property
    def name(self) -> str:
        return "emulate_device"

    @property
    def description(self) -> str:
        return (
            "Emulate a specific device viewport. "
            "Available devices: iphone_12, iphone_14, pixel_5, ipad, ipad_pro, desktop_hd, desktop_4k. "
            "Note: User agent cannot be changed on existing session."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="device_name",
                type="string",
                description="Device to emulate (e.g., 'iphone_14', 'pixel_5', 'ipad')",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        device_name: str,
    ) -> str:
        """Emulate device."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.emulate_device(device_name)

            if result.get("success"):
                viewport = result["viewport"]
                mobile = "mobile" if result.get("is_mobile") else "desktop"
                lines = [
                    f"Emulating {result['device']} ({mobile})",
                    f"Viewport: {viewport['width']}x{viewport['height']}",
                ]
                if result.get("note"):
                    lines.append(f"Note: {result['note']}")
                return "\n".join(lines)
            else:
                error = result.get("error", "Unknown error")
                if result.get("available_devices"):
                    error += f"\nAvailable: {', '.join(result['available_devices'])}"
                return f"Emulate device failed: {error}"

        except Exception as e:
            logger.error(f"Emulate device error: {e}", exc_info=True)
            return f"Emulate device error: {str(e)}"


class SetGeolocationTool(Tool):
    """Set the browser's geolocation."""

    @property
    def name(self) -> str:
        return "set_geolocation"

    @property
    def description(self) -> str:
        return (
            "Set the browser's geolocation coordinates. "
            "Useful for testing location-based features."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="latitude",
                type="number",
                description="Latitude (-90 to 90)",
                required=True,
            ),
            ToolParameter(
                name="longitude",
                type="number",
                description="Longitude (-180 to 180)",
                required=True,
            ),
            ToolParameter(
                name="accuracy",
                type="number",
                description="Accuracy in meters (default: 100)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        latitude: float,
        longitude: float,
        accuracy: float = 100,
    ) -> str:
        """Set geolocation."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        if not -90 <= latitude <= 90:
            return "Error: Latitude must be between -90 and 90"
        if not -180 <= longitude <= 180:
            return "Error: Longitude must be between -180 and 180"

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_geolocation(latitude, longitude, accuracy)

            if result.get("success"):
                geo = result["geolocation"]
                return (
                    f"Geolocation set:\n"
                    f"  Latitude: {geo['latitude']}\n"
                    f"  Longitude: {geo['longitude']}\n"
                    f"  Accuracy: {geo['accuracy']}m"
                )
            else:
                return f"Set geolocation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set geolocation error: {e}", exc_info=True)
            return f"Set geolocation error: {str(e)}"


class SetTimezoneTool(Tool):
    """Set the browser's timezone."""

    @property
    def name(self) -> str:
        return "set_timezone"

    @property
    def description(self) -> str:
        return (
            "Set the browser's timezone. "
            "Use IANA timezone IDs like 'America/New_York', 'Europe/London', 'Asia/Tokyo'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="timezone_id",
                type="string",
                description="IANA timezone ID (e.g., 'America/New_York', 'UTC')",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        timezone_id: str,
    ) -> str:
        """Set timezone."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_timezone(timezone_id)

            if result.get("success"):
                return f"Timezone set to: {result['timezone']}"
            else:
                return f"Set timezone failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set timezone error: {e}", exc_info=True)
            return f"Set timezone error: {str(e)}"
