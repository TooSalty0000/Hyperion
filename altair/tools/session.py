"""Session management tools for Altair."""

from typing import Optional, List
import logging

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class CreateSessionTool(Tool):
    """Create a new terminal session with Discord channel."""

    @property
    def name(self) -> str:
        return "create_session"

    @property
    def description(self) -> str:
        return (
            "Create a new terminal session with its own Discord channel. "
            "Use the 'project' parameter to specify a project name - the session "
            "will be created in that project's workspace directory. "
            "Optionally auto-start Claude Code after session creation."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="project",
                type="string",
                description="Project name to load workspace from. The session will be created in the project's directory.",
                required=False
            ),
            ToolParameter(
                name="command",
                type="string",
                description="Command to run (default: shell)",
                required=False
            ),
            ToolParameter(
                name="workspace_dir",
                type="string",
                description="Working directory path (overrides project path if both specified)",
                required=False
            ),
            ToolParameter(
                name="start_claude_code",
                type="boolean",
                description="Automatically start Claude Code after creating the session. Default: False",
                required=False
            ),
            ToolParameter(
                name="continue_claude_session",
                type="boolean",
                description="If start_claude_code=True, use --continue flag for existing conversation. Default: False (fresh start)",
                required=False
            )
        ]

    async def execute(
        self,
        context: ToolContext,
        project: Optional[str] = None,
        command: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        start_claude_code: bool = False,
        continue_claude_session: bool = False
    ) -> str:
        # Validate we have the components needed for channel creation
        if not context.channel_manager:
            return "Error: Channel manager not available. Cannot create session channel."
        if not context.discord_bot:
            return "Error: Discord bot not available. Cannot create session channel."
        if not context.current_guild_id:
            return "Error: No guild context. Sessions can only be created in Discord servers."

        # Resolve workspace from project if specified
        project_info = None
        project_name = None
        if project and context.project_manager:
            project_info = await context.project_manager.get(project)
            if project_info:
                workspace_dir = workspace_dir or project_info.path
                project_name = project_info.name
                if not command and project_info.settings:
                    command = project_info.settings.get("default_command")
            else:
                return f"Error: Project '{project}' not found. Use list_projects to see available projects."

        # Check if a session already exists for this project
        if project_name:
            existing_session = await context.session_registry.get_session_by_project(project_name)
            if existing_session and existing_session.is_alive:
                # Session already exists and is running
                channel = context.discord_bot.get_channel(existing_session.channel_id)
                channel_ref = f"<#{channel.id}>" if channel else "(channel not found)"
                return (
                    f"✅ Session for project '{project_name}' already exists and is running.\n"
                    f"Session: #{existing_session.session_id}\n"
                    f"Channel: {channel_ref}\n"
                    f"PID: {existing_session.data.pid}\n"
                    f"Use the existing session instead of creating a new one."
                )
            elif existing_session:
                # Session exists but process is dead - clean it up
                await context.session_registry.terminate_session(existing_session.session_id)
                if existing_session.channel_id and context.channel_manager:
                    await context.channel_manager.delete_channel(existing_session.channel_id)

        # Get the guild from the bot
        guild = context.discord_bot.get_guild(context.current_guild_id)
        if not guild:
            return f"Error: Could not find guild with ID {context.current_guild_id}"

        # Get next session ID and create channel
        next_id = await context.session_registry.get_next_id()
        channel = await context.channel_manager.create_channel(guild, next_id)
        if not channel:
            return "Error: Failed to create Discord channel for session."

        try:
            # Create the session with the new channel's ID
            session = await context.session_registry.create_session(
                channel_id=channel.id,
                command=command,
                workspace_dir=workspace_dir,
                project_name=project_name
            )

            # Start output loop if callback is available
            if context.start_output_loop_callback:
                context.start_output_loop_callback(session)

            result_parts = [
                f"✅ Created session #{session.session_id}",
                f"Channel: <#{channel.id}>"
            ]

            if workspace_dir:
                result_parts.append(f"Workspace: {workspace_dir}")
            if project:
                result_parts.append(f"Project: {project}")
            if command:
                result_parts.append(f"Command: {command}")

            attach_cmd = session.get_attach_command()
            if attach_cmd:
                result_parts.append(f"Local attach: `{attach_cmd}`")

            # Auto-start Claude Code if requested
            if start_claude_code:
                import asyncio
                # Wait a moment for the shell to start
                await asyncio.sleep(0.5)

                claude_cmd = "claude"
                if continue_claude_session:
                    claude_cmd += " --continue"

                await session.terminal.send_input(claude_cmd + "\n")
                await context.session_registry.update_mode(session.session_id, "claude_code")
                result_parts.append(f"Claude Code: Started ({'continuing' if continue_claude_session else 'fresh session'})")

            return "\n".join(result_parts)

        except Exception as e:
            # Clean up channel on failure
            await context.channel_manager.delete_channel(channel.id)
            return f"Error creating session: {e}"


class TerminateSessionTool(Tool):
    """Terminate a session and its Discord channel."""

    @property
    def name(self) -> str:
        return "terminate_session"

    @property
    def description(self) -> str:
        return (
            "Terminate a terminal session, kill the process, and delete its Discord channel. "
            "This fully cleans up the session."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to terminate. Uses current session if not specified.",
                required=False
            )
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: Optional[int] = None
    ) -> str:
        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session ID provided and no current session."

        # Get session to find channel_id before terminating
        session = await context.session_registry.get_session(sid)
        if not session:
            return f"Error: Session #{sid} not found."

        channel_id = session.channel_id

        # Terminate the session
        success = await context.session_registry.terminate_session(sid)
        if not success:
            return f"Error: Failed to terminate session #{sid}."

        # Delete the channel
        if context.channel_manager and channel_id:
            await context.channel_manager.delete_channel(channel_id)

        return f"✅ Session #{sid} terminated and channel deleted."


class SwitchSessionTool(Tool):
    """Switch to a different session."""

    @property
    def name(self) -> str:
        return "switch_session"

    @property
    def description(self) -> str:
        return (
            "Switch the current active session context. "
            "Subsequent commands without a session_id will use this session."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to switch to",
                required=True
            )
        ]

    async def execute(self, context: ToolContext, session_id: int) -> str:
        session = await context.session_registry.get_session(session_id)
        if not session:
            return f"Error: Session #{session_id} not found."

        status = "running" if session.is_alive else "stopped"
        channel_info = f"<#{session.channel_id}>" if session.channel_id else "unknown"
        return f"Switched to session #{session_id} [{status}] in {channel_info}."


class SetSessionNoteTool(Tool):
    """Set notes for a session to remember what it's for."""

    @property
    def name(self) -> str:
        return "set_session_note"

    @property
    def description(self) -> str:
        return (
            "Set a note for a session to remember what it's being used for. "
            "Use this to track which session is for which task, project, or purpose. "
            "This helps you avoid creating duplicate sessions for the same work."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="note",
                type="string",
                description="The note describing what this session is for (e.g., 'Building React dashboard for Project X')",
                required=True,
            ),
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to annotate. Uses current session if not specified.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        note: str,
        session_id: Optional[int] = None,
    ) -> str:
        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session ID provided and no current session."

        session = await context.session_registry.get_session(sid)
        if not session:
            return f"Error: Session #{sid} not found."

        # Update the note in session data
        await context.session_registry.store.update(sid, notes=note)
        session.data.notes = note

        return f"✅ Note set for session #{sid}: {note}"


class GetSessionNoteTool(Tool):
    """Get the note for a session."""

    @property
    def name(self) -> str:
        return "get_session_note"

    @property
    def description(self) -> str:
        return "Get the note for a session to see what it's being used for."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to get note for. Uses current session if not specified.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: Optional[int] = None,
    ) -> str:
        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session ID provided and no current session."

        session = await context.session_registry.get_session(sid)
        if not session:
            return f"Error: Session #{sid} not found."

        note = session.data.notes
        if note:
            return f"Session #{sid} note: {note}"
        else:
            return f"Session #{sid} has no note set."


class ListProjectsTool(Tool):
    """List available projects."""

    @property
    def name(self) -> str:
        return "list_projects"

    @property
    def description(self) -> str:
        return "List all registered projects and their workspace directories."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        if not context.project_manager:
            return "Error: Project manager not available."

        projects = await context.project_manager.list_all()
        if not projects:
            return "No projects registered."

        lines = ["**Registered Projects:**"]
        for p in projects:
            lines.append(f"- **{p.name}**: `{p.path}`")
            if p.settings:
                if "default_command" in p.settings:
                    lines.append(f"  Default command: `{p.settings['default_command']}`")

        return "\n".join(lines)
