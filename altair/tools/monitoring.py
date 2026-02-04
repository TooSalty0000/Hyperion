"""Session monitoring tools for Altair - Active feedback and stuck detection."""

import re
import asyncio
import logging
from typing import Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Detected state of a CLI session."""
    WORKING = "working"           # Active output, making progress
    WAITING_INPUT = "waiting"     # Waiting for user input/permission
    IDLE = "idle"                 # No recent activity
    COMPLETED = "completed"       # Task finished successfully
    ERROR = "error"               # Error occurred
    UNKNOWN = "unknown"           # Cannot determine state


@dataclass
class MonitorResult:
    """Result of monitoring a session."""
    state: SessionState
    message: str
    details: Optional[str] = None
    prompt_detected: Optional[str] = None  # The prompt/question if waiting
    recent_output: Optional[str] = None    # Last few lines of output


class MonitorSessionTool(Tool):
    """
    Actively monitor a CLI session and report its current state.

    This tool checks the session output and detects:
    - Whether the CLI is working on something
    - Whether it's waiting for user input (permission prompts)
    - Whether it appears stuck or idle
    - Whether it has completed or errored

    Use this in a loop to actively track progress and provide feedback.
    """

    # Patterns for detecting permission/input prompts
    WAITING_PATTERNS = [
        # Claude Code permission prompts
        (r'Do you want to proceed\?', "Claude Code is asking for permission to proceed"),
        (r'Allow this action\?', "Claude Code is asking to allow an action"),
        (r'y/n\)?[\s:]*$', "Waiting for yes/no confirmation"),
        (r'\[Y/n\][\s:]*$', "Waiting for confirmation (default Yes)"),
        (r'\[y/N\][\s:]*$', "Waiting for confirmation (default No)"),
        (r'Press Enter to continue', "Waiting for Enter key"),
        (r'Type .+ to confirm', "Waiting for typed confirmation"),
        (r'❯\s*\d+\.\s+Yes', "Claude Code showing permission options"),
        (r'Esc to cancel', "Claude Code permission dialog active"),
        # Generic input prompts
        (r':\s*$', "May be waiting for input"),
        (r'\?\s*$', "Question prompt detected"),
    ]

    # Patterns for detecting completion
    COMPLETION_PATTERNS = [
        (r'✓\s*(?:Done|Complete|Finished)', "Task completed successfully"),
        (r'Successfully\s+(?:created|updated|deleted|installed)', "Operation succeeded"),
        (r'\$\s*$', "Returned to shell prompt"),
        (r'claude>\s*$', "Returned to Claude prompt"),
        (r'Task completed', "Task completed"),
    ]

    # Patterns for detecting errors
    ERROR_PATTERNS = [
        (r'Error:', "Error detected"),
        (r'Failed:', "Failure detected"),
        (r'Exception:', "Exception occurred"),
        (r'fatal:', "Fatal error"),
        (r'ENOENT|EACCES|EPERM', "File system error"),
        (r'command not found', "Command not found"),
        (r'Permission denied', "Permission denied"),
    ]

    # Patterns indicating active work
    WORKING_PATTERNS = [
        (r'⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏', "Spinner detected - working"),
        (r'\.{3,}$', "Progress dots - working"),
        (r'\d+%', "Progress percentage - working"),
        (r'Reading|Writing|Processing|Analyzing|Building|Installing', "Active operation"),
        (r'⏺\s+\w+\(', "Claude Code tool execution"),
    ]

    @property
    def name(self) -> str:
        return "monitor_session"

    @property
    def description(self) -> str:
        return (
            "Monitor a CLI session and report its current state. "
            "Detects if Claude Code is working, waiting for input, stuck, or completed. "
            "Use this to actively track progress and detect when action is needed. "
            "Returns a status report you can relay to the user."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to monitor. Uses current session if not specified.",
                required=False
            ),
            ToolParameter(
                name="watch_seconds",
                type="integer",
                description="How long to watch for changes (default 5 seconds). Longer = more accurate state detection.",
                required=False
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: Optional[int] = None,
        watch_seconds: int = 5
    ) -> str:
        """Monitor session and return status."""
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
                return f"⚠️ Session #{sid} is not running (process may have exited)."

            # Get initial output
            initial_output = await session.terminal.get_output() or ""
            initial_lines = initial_output.split('\n')

            # Watch for changes (cancellation-aware)
            ct = context.cancellation_token
            if ct:
                await ct.sleep(watch_seconds)
            else:
                await asyncio.sleep(watch_seconds)

            # Get current output
            current_output = await session.terminal.get_output() or ""
            current_lines = current_output.split('\n')

            # Analyze the recent output (last 20 lines)
            recent = '\n'.join(current_lines[-20:]) if current_lines else ""

            # Check if output changed during watch period
            output_changed = current_output != initial_output

            # Detect state
            result = self._analyze_output(recent, output_changed)

            # Format response
            response_parts = [f"**Session #{sid} Status:** {result.state.value.upper()}"]
            response_parts.append(f"📋 {result.message}")

            if result.prompt_detected:
                response_parts.append(f"⚠️ **Prompt detected:** {result.prompt_detected}")

            if result.details:
                response_parts.append(f"ℹ️ {result.details}")

            # Add recent output preview
            if recent:
                preview_lines = recent.strip().split('\n')[-5:]
                preview = '\n'.join(preview_lines)
                if preview:
                    response_parts.append(f"\n**Recent output:**\n```\n{preview}\n```")

            return '\n'.join(response_parts)

        except Exception as e:
            logger.error(f"Error monitoring session: {e}", exc_info=True)
            return f"Error monitoring session: {str(e)}"

    def _analyze_output(self, output: str, output_changed: bool) -> MonitorResult:
        """Analyze output to determine session state."""

        # Check for waiting/input prompts first (highest priority)
        for pattern, description in self.WAITING_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return MonitorResult(
                    state=SessionState.WAITING_INPUT,
                    message="Claude Code is waiting for input or permission",
                    prompt_detected=description,
                    recent_output=output[-500:]
                )

        # Check for errors
        for pattern, description in self.ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return MonitorResult(
                    state=SessionState.ERROR,
                    message=description,
                    details="Check the output for error details",
                    recent_output=output[-500:]
                )

        # Check for completion
        for pattern, description in self.COMPLETION_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return MonitorResult(
                    state=SessionState.COMPLETED,
                    message=description,
                    recent_output=output[-500:]
                )

        # Check for active work
        for pattern, description in self.WORKING_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return MonitorResult(
                    state=SessionState.WORKING,
                    message=description,
                    details="Output is changing" if output_changed else "Active indicator detected",
                    recent_output=output[-500:]
                )

        # If output changed, assume working
        if output_changed:
            return MonitorResult(
                state=SessionState.WORKING,
                message="Output is changing - work in progress",
                recent_output=output[-500:]
            )

        # No changes and no patterns - likely idle
        return MonitorResult(
            state=SessionState.IDLE,
            message="No recent activity detected",
            details="The session may be waiting or the process may have finished",
            recent_output=output[-500:]
        )


class CheckForPromptTool(Tool):
    """
    Quick check if Claude Code is waiting for user input/permission.

    Faster than full monitor - just checks for permission prompts.
    Returns the prompt if found, or empty if not waiting.
    """

    PROMPT_PATTERNS = [
        r'Do you want to proceed\?',
        r'Allow this action\?',
        r'❯\s*\d+\.\s+Yes',
        r'Esc to cancel',
        r'\[Y/n\]',
        r'\[y/N\]',
        r'y/n\)?[\s:]*$',
    ]

    @property
    def name(self) -> str:
        return "check_for_prompt"

    @property
    def description(self) -> str:
        return (
            "Quick check if Claude Code is waiting for permission or user input. "
            "Returns the prompt text if found, or indicates no prompt detected. "
            "Use this for fast polling between monitor_session calls."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                type="integer",
                description="Session ID to check. Uses current session if not specified.",
                required=False
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        session_id: Optional[int] = None
    ) -> str:
        """Check for prompt."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            output = await session.terminal.get_output() or ""
            recent = '\n'.join(output.split('\n')[-15:])

            for pattern in self.PROMPT_PATTERNS:
                match = re.search(pattern, recent, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Get surrounding context
                    lines = recent.split('\n')
                    prompt_lines = [l for l in lines[-10:] if l.strip()]
                    return f"⚠️ **Permission prompt detected:**\n```\n{chr(10).join(prompt_lines[-5:])}\n```"

            return "✅ No permission prompt detected - Claude Code appears to be working or idle."

        except Exception as e:
            return f"Error: {str(e)}"


class AnswerTerminalPromptTool(Tool):
    """
    Send a response to a CLI/terminal prompt (e.g., Claude Code permission dialogs).

    IMPORTANT: This tool sends keystrokes to the TERMINAL, not to Discord.
    - Use this ONLY when monitor_session or check_for_prompt detects a terminal prompt
    - This is for answering CLI permission dialogs like "Do you want to proceed? [Y/n]"
    - This is NOT for responding to the user in Discord chat
    - To reply to users, just return text in your response - don't use this tool
    """

    @property
    def name(self) -> str:
        return "answer_terminal_prompt"

    @property
    def description(self) -> str:
        return (
            "Send a response to a TERMINAL/CLI prompt (like Claude Code's permission dialogs). "
            "This sends keystrokes to the terminal session, NOT to Discord. "
            "Use 'yes' to approve, 'no' to reject, or a number to select an option. "
            "ONLY use this when monitor_session shows WAITING state with a terminal prompt detected. "
            "Do NOT use this to reply to users - just include your reply in your response text."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="response",
                type="string",
                description=(
                    "Response to send to the TERMINAL: 'yes'/'y' to approve, 'no'/'n' to reject, "
                    "'enter' to press Enter, 'escape' to cancel, or a number (1,2,3) to select option. "
                    "This is sent as keystrokes to the CLI, not as a Discord message."
                ),
                required=True
            ),
            ToolParameter(
                name="session_id",
                type="integer",
                description="Terminal session ID. Uses current session if not specified.",
                required=False
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        response: str,
        session_id: Optional[int] = None
    ) -> str:
        """Send response to prompt."""
        if not context.session_registry:
            return "Error: Session registry not available."

        sid = session_id or context.current_session_id
        if not sid:
            return "Error: No session specified."

        try:
            session = await context.session_registry.get_session(sid)
            if not session:
                return f"Error: Session #{sid} not found."

            if not session.is_alive:
                return f"Error: Session #{sid} is not running."

            # Map common responses
            # Claude Code uses TUI-style selectors - navigate with arrows, select with Enter
            response_lower = response.lower().strip()

            if response_lower == 'yes' or response_lower == 'y' or response_lower == '1':
                # First option (Yes) is typically already selected, just press Enter
                await session.terminal.send_control('enter')
                return "✅ Sent approval (Enter) to session - selected first option"
            elif response_lower == 'no' or response_lower == 'n' or response_lower == '2':
                # Second option - press Down arrow to navigate, then Enter to select
                await session.terminal.send_control('down')
                await asyncio.sleep(0.1)  # Small delay for UI to update
                await session.terminal.send_control('enter')
                return "❌ Sent rejection (Down + Enter) to session - selected second option"
            elif response_lower == 'escape' or response_lower == 'esc' or response_lower == 'cancel':
                await session.terminal.send_control('esc')
                return "🚫 Sent Escape to cancel"
            elif response_lower == 'enter':
                await session.terminal.send_control('enter')
                return "⏎ Sent Enter key to session"
            elif response_lower == 'up':
                await session.terminal.send_control('up')
                return "⬆️ Sent Up arrow to session"
            elif response_lower == 'down':
                await session.terminal.send_control('down')
                return "⬇️ Sent Down arrow to session"
            else:
                # Try to parse as a number (option index)
                try:
                    option_num = int(response_lower)
                    # Navigate down (option_num - 1) times from first option
                    for _ in range(option_num - 1):
                        await session.terminal.send_control('down')
                        await asyncio.sleep(0.05)
                    await session.terminal.send_control('enter')
                    return f"✅ Selected option {option_num} (navigated + Enter)"
                except ValueError:
                    # Send as custom text input
                    await session.terminal.send_input(response + "\n")
                    return f"📝 Sent custom response: {response[:50]}..."

        except Exception as e:
            return f"Error: {str(e)}"
