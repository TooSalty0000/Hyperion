"""Project summary tools for Vega."""

import logging
from typing import Optional, List, TYPE_CHECKING

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from shared.project import ProjectManager

logger = logging.getLogger(__name__)


class GetProjectSummaryTool(Tool):
    """
    Get a summary of all registered projects and their status.

    This tool provides Vega with an overview of projects without
    needing to query Altair directly for simple status checks.
    """

    def __init__(self, project_manager: Optional['ProjectManager'] = None):
        self._project_manager = project_manager

    @property
    def name(self) -> str:
        return "get_project_summary"

    @property
    def description(self) -> str:
        return (
            "Get a summary of all registered projects including their names, "
            "paths, and current status. Use this to answer questions about "
            "project overview. For detailed status, @mention Altair."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="project_name",
                type="string",
                description="Optional: Get summary for a specific project only",
                required=False,
            )
        ]

    async def execute(self, context: ToolContext, project_name: str = None) -> str:
        """Get project summary."""
        if not self._project_manager:
            return "Project management is not available. Configure REDIS_URL to enable."

        try:
            if project_name:
                # Get specific project
                project = await self._project_manager.get(project_name)
                if not project:
                    return f"Project '{project_name}' not found."

                settings = project.settings or {}
                return (
                    f"Project: {project.name}\n"
                    f"Path: {project.path}\n"
                    f"Default Command: {settings.get('default_command', 'Not set')}\n"
                    f"Status: {settings.get('status', 'Unknown')}"
                )

            # Get all projects
            projects = await self._project_manager.list_all()
            if not projects:
                return "No projects registered. Users can add projects with !vp <path>."

            lines = ["Registered Projects:"]
            for p in projects:
                status = p.settings.get('status', 'idle') if p.settings else 'idle'
                lines.append(f"- {p.name}: {p.path} (status: {status})")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error getting project summary: {e}", exc_info=True)
            return f"Error retrieving project information: {str(e)}"
