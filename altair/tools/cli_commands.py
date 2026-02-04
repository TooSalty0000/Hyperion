"""CLI interface tools for Altair to interact with terminal sessions."""

import re
import asyncio
import logging
from typing import Optional, List, TYPE_CHECKING

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from shared.session import Session

logger = logging.getLogger(__name__)


class StartClaudeCodeTool(Tool):
    """
    Start Claude Code CLI in a session.

    By default starts fresh (no --continue).
    Use continue_session=True to resume an existing conversation.
    """

    @property
    def name(self) -> str:
        return "start_claude_code"

    @property
    def description(self) -> str:
        return (
            "Start Claude Code CLI in a terminal session. "
            "By default starts fresh. Use continue_session=True to resume "
            "an existing conversation with 'claude --continue'."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="The session ID to start Claude Code in. If not provided, uses current session.",
                required=False,
            ),
            ToolParameter(
                name="working_directory",
                type="string",
                description="Optional directory to start Claude Code in.",
                required=False,
            ),
            ToolParameter(
                name="continue_session",
                type="boolean",
                description="Resume existing conversation with --continue. Default: False (fresh start).",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: int = None,
        working_directory: str = None,
        continue_session: bool = False,
    ) -> str:
        """Start Claude Code in session."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified and no current session set."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            if not session.is_alive:
                return f"Error: Session #{sid} is not running."

            # Build command - fresh start by default
            cmd = "claude"
            if continue_session:
                cmd = "claude --continue"

            if working_directory:
                cmd = f"cd {working_directory} && {cmd}"

            # Send command with enter
            await session.terminal.send_input(cmd + "\n")

            mode = "resuming existing conversation" if continue_session else "starting fresh"
            return f"Started Claude Code in session #{sid} ({mode}). Claude is ready to receive instructions."

        except Exception as e:
            logger.error(f"Error starting Claude Code: {e}", exc_info=True)
            return f"Error starting Claude Code: {str(e)}"


class SendCLICommandTool(Tool):
    """
    Send a command or instruction to a CLI tool in a session.

    Automatically appends Enter key to the command.
    Requests user permission for dangerous commands.
    """

    # Patterns that indicate dangerous commands requiring permission
    DANGEROUS_PATTERNS = [
        # Destructive operations
        (r'\brm\s+-rf?\b', "delete files/directories"),
        (r'\brmdir\b', "delete directories"),
        (r'\bdrop\s+(?:table|database)\b', "drop database objects"),
        (r'\btruncate\b', "truncate data"),
        (r'\bdelete\s+from\b', "delete database records"),
        # Git operations that modify remote
        (r'\bgit\s+push\b', "push to remote repository"),
        (r'\bgit\s+push\s+.*--force\b', "force push (dangerous!)"),
        (r'\bgit\s+reset\s+--hard\b', "hard reset"),
        # System modifications
        (r'\bsudo\b', "run with elevated privileges"),
        (r'\bchmod\b', "change file permissions"),
        (r'\bchown\b', "change file ownership"),
        # Deployment/production
        (r'\bdeploy\b', "deploy"),
        (r'\bproduction\b', "production environment"),
        # Package management that installs
        (r'\bnpm\s+publish\b', "publish package"),
        (r'\bpip\s+install\b', "install packages"),
        (r'\bnpm\s+install\b', "install packages"),
        # Network/security
        (r'\bcurl\s+.*\|\s*(?:bash|sh)\b', "pipe remote script to shell"),
        (r'\bwget\s+.*\|\s*(?:bash|sh)\b', "pipe remote script to shell"),
    ]

    @property
    def name(self) -> str:
        return "send_cli_command"

    @property
    def description(self) -> str:
        return (
            "Send a command or instruction to the CLI tool running in a session. "
            "Use this to instruct claude code, gemini, or other agentic CLI tools "
            "to perform work. The command is sent as text input followed by Enter. "
            "Dangerous commands will require user approval first."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The command or instruction to send to the CLI tool",
                required=True,
            ),
            ToolParameter(
                name="session_id",
                type="integer",
                description="The session ID to send the command to. If not provided, uses current session.",
                required=False,
            ),
        ]

    def _check_dangerous(self, command: str) -> tuple[bool, str]:
        """Check if command matches dangerous patterns. Returns (is_dangerous, reason)."""
        command_lower = command.lower()
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return True, reason
        return False, ""

    async def execute(
        self,
        context: ToolContext,
        command: str,
        session_id: int = None
    ) -> str:
        """Send command to CLI tool."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified and no current session set."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            if not session.is_alive:
                return f"Error: Session #{sid} is not running."

            # Check for dangerous commands
            is_dangerous, reason = self._check_dangerous(command)
            if is_dangerous and context.permission_manager and context.current_channel_id:
                # Request permission from user
                channel = context.discord_bot.get_channel(context.current_channel_id)
                if channel:
                    from altair.permission import ApprovalStatus
                    status = await context.permission_manager.request_permission(
                        channel=channel,
                        action_type="cli_command",
                        description=f"Potentially dangerous: {reason}",
                        command=command,
                        session_id=sid,
                    )

                    if status == ApprovalStatus.APPROVED:
                        logger.info(f"User approved dangerous command: {command[:50]}")
                    elif status == ApprovalStatus.REJECTED:
                        return f"❌ Command rejected by user. Reason flagged: {reason}"
                    elif status == ApprovalStatus.TIMEOUT:
                        return f"⏱️ Permission request timed out. Command not sent. Reason flagged: {reason}"
                    else:
                        return f"⚠️ Permission request failed. Command not sent."

            # Send the command with Enter key
            if not command.endswith('\n'):
                command = command + '\n'
            await session.terminal.send_input(command)

            return f"Command sent to session #{sid}: {command[:100].strip()}{'...' if len(command) > 100 else ''}"

        except Exception as e:
            logger.error(f"Error sending CLI command: {e}", exc_info=True)
            return f"Error sending command: {str(e)}"


class GetCLIOutputTool(Tool):
    """
    Get the current output from a CLI session.

    Allows Altair to see what the CLI tool has produced.
    """

    @property
    def name(self) -> str:
        return "get_cli_output"

    @property
    def description(self) -> str:
        return (
            "Get the current terminal output from a CLI session. "
            "Use this to see what the CLI tool has produced, check for errors, "
            "or understand the current state of work."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="The session ID to get output from. If not provided, uses current session.",
                required=False,
            ),
            ToolParameter(
                name="lines",
                type="integer",
                description="Number of lines to retrieve (from the end). Default 50.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: int = None,
        lines: int = 50
    ) -> str:
        """Get CLI output."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified and no current session set."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            # Get output from terminal
            output = await session.terminal.get_output()
            if not output:
                return f"Session #{sid}: No output available."

            # Get last N lines
            output_lines = output.split('\n')
            if len(output_lines) > lines:
                output_lines = output_lines[-lines:]

            result = '\n'.join(output_lines)
            return f"Session #{sid} output (last {min(lines, len(output_lines))} lines):\n{result}"

        except Exception as e:
            logger.error(f"Error getting CLI output: {e}", exc_info=True)
            return f"Error getting output: {str(e)}"


class WaitForCompletionTool(Tool):
    """
    Wait for a CLI operation to complete.

    Monitors the terminal output for completion indicators.
    """

    # Patterns that indicate a CLI tool might be done
    COMPLETION_PATTERNS = [
        r'(?:claude|gemini|gpt)>?\s*$',  # CLI prompt returned
        r'\$ $',  # Shell prompt
        r'>>> $',  # Python REPL
        r'Done\.?$',
        r'Complete\.?$',
        r'Finished\.?$',
        r'Error:',
        r'Failed:',
    ]

    @property
    def name(self) -> str:
        return "wait_for_completion"

    @property
    def description(self) -> str:
        return (
            "Wait for an operation in a CLI session to complete. "
            "Monitors the terminal output for completion indicators like "
            "prompts returning or 'Done'/'Complete' messages. "
            "Use after sending a command that takes time to execute."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="The session ID to monitor. If not provided, uses current session.",
                required=False,
            ),
            ToolParameter(
                name="timeout_seconds",
                type="integer",
                description="Maximum time to wait in seconds. Default 300.",
                required=False,
            ),
            ToolParameter(
                name="completion_pattern",
                type="string",
                description="Optional regex pattern to detect completion.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: int = None,
        timeout_seconds: int = 300,
        completion_pattern: str = None
    ) -> str:
        """Wait for completion."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified and no current session set."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            if not session.is_alive:
                return f"Session #{sid} is not running."

            # Build patterns to check
            patterns = self.COMPLETION_PATTERNS.copy()
            if completion_pattern:
                patterns.insert(0, completion_pattern)

            compiled_patterns = [re.compile(p, re.MULTILINE) for p in patterns]

            # Poll for completion
            start_time = asyncio.get_event_loop().time()
            last_output = ""

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout_seconds:
                    return (
                        f"Timeout after {timeout_seconds}s waiting for completion. "
                        f"The operation may still be running."
                    )

                output = await session.terminal.get_output()
                if output and output != last_output:
                    last_output = output
                    # Check for completion patterns in last few lines
                    recent = '\n'.join(output.split('\n')[-10:])
                    for pattern in compiled_patterns:
                        if pattern.search(recent):
                            return f"Operation completed. Pattern detected: {pattern.pattern}"

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error waiting for completion: {e}", exc_info=True)
            return f"Error waiting: {str(e)}"


class GetSessionStatusTool(Tool):
    """Get the status of a CLI session."""

    @property
    def name(self) -> str:
        return "get_session_status"

    @property
    def description(self) -> str:
        return (
            "Get the status of a CLI session including whether it's alive, "
            "the process ID, and terminal size. Use to check on session health."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="The session ID to check. If not provided, uses current session.",
                required=False,
            ),
        ]

    async def execute(self, context: ToolContext, session_id: int = None) -> str:
        """Get session status."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified and no current session set."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            status = "Running" if session.is_alive else "Stopped"
            pid = session.data.pid
            backend = session.terminal.get_name()
            config = session.terminal.config
            channel_ref = f"<#{session.channel_id}>" if session.channel_id else "(no channel)"

            return (
                f"Session #{sid} Status:\n"
                f"- Status: {status}\n"
                f"- Channel: {channel_ref}\n"
                f"- PID: {pid}\n"
                f"- Backend: {backend}\n"
                f"- Terminal Size: {config.cols}x{config.rows}"
            )

        except Exception as e:
            logger.error(f"Error getting session status: {e}", exc_info=True)
            return f"Error: {str(e)}"


class ListActiveSessionsTool(Tool):
    """List all active CLI sessions."""

    @property
    def name(self) -> str:
        return "list_active_sessions"

    @property
    def description(self) -> str:
        return (
            "List all active CLI sessions with their IDs, status, and channels. "
            "Use to find available sessions for work or check what's running."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []  # No parameters needed

    async def execute(self, context: ToolContext) -> str:
        """List active sessions."""
        if not context.session_registry:
            return "Error: Session registry not available."

        try:
            sessions = await context.session_registry.list_sessions()
            if not sessions:
                return "No active sessions. Use create_session to start one."

            lines = ["**Active CLI Sessions:**"]
            for session in sessions:
                status = "Running" if session.is_alive else "Stopped"
                channel_ref = f"<#{session.channel_id}>" if session.channel_id else "(no channel)"
                note_str = f"\n  📝 {session.data.notes}" if session.data.notes else ""
                lines.append(
                    f"- Session #{session.session_id}: [{status}] {channel_ref} (PID: {session.data.pid}){note_str}"
                )

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            return f"Error: {str(e)}"
