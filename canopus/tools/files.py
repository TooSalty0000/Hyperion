"""File upload and download tools for Canopus browser automation."""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class UploadFileTool(Tool):
    """Upload a file to a file input element."""

    @property
    def name(self) -> str:
        return "upload_file"

    @property
    def description(self) -> str:
        return (
            "Upload a file to a file input element on the page. "
            "Provide the CSS selector for the file input and the path to the file. "
            "The file must exist on the local filesystem."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector for the file input element (e.g., 'input[type=\"file\"]')",
                required=True,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to upload",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        selector: str,
        file_path: str,
    ) -> str:
        """Upload file."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.upload_file(selector, file_path)

            if result.get("success"):
                return (
                    f"File uploaded successfully.\n"
                    f"File: {result['file']}\n"
                    f"Size: {result['size']} bytes"
                )
            else:
                return f"Upload failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Upload file error: {e}", exc_info=True)
            return f"Upload file error: {str(e)}"


class SetDownloadPathTool(Tool):
    """Set the download directory for file downloads."""

    @property
    def name(self) -> str:
        return "set_download_path"

    @property
    def description(self) -> str:
        return (
            "Set the directory where downloaded files will be saved. "
            "Call this before triggering a download."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path for downloads (will be created if doesn't exist)",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, path: str) -> str:
        """Set download path."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.set_download_path(path)

            if result.get("success"):
                return f"Download path set to: {result['download_path']}"
            else:
                return f"Set download path failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Set download path error: {e}", exc_info=True)
            return f"Set download path error: {str(e)}"


class WaitForDownloadTool(Tool):
    """Wait for a file download to complete."""

    @property
    def name(self) -> str:
        return "wait_for_download"

    @property
    def description(self) -> str:
        return (
            "Wait for a file download to complete after triggering a download action. "
            "Returns information about the downloaded file. "
            "Set download path first with set_download_path for files to be saved."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="timeout",
                type="integer",
                description="Maximum time to wait in milliseconds (default: 30000)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        timeout: int = 30000,
    ) -> str:
        """Wait for download."""
        if not hasattr(context, 'browser_manager') or not context.browser_manager:
            return "Error: Browser manager not available."

        if not context.current_channel_id:
            return "Error: No channel context for browser session."

        try:
            session = await context.browser_manager.get(context.current_channel_id)
            if not session:
                return "Error: No active browser session. Use navigate_to first."

            result = await session.wait_for_download(timeout)

            if result.get("success"):
                lines = [
                    "Download completed.",
                    f"Filename: {result['filename']}",
                    f"URL: {result['url']}",
                ]
                if result.get("saved_path"):
                    lines.append(f"Saved to: {result['saved_path']}")
                return "\n".join(lines)
            else:
                return f"Download failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Wait for download error: {e}", exc_info=True)
            return f"Wait for download error: {str(e)}"
