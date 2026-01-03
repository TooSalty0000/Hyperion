"""Permission system for Altair - asks user before executing sensitive actions."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import discord

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of a permission request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class PermissionRequest:
    """A request for permission to perform an action."""
    action_type: str  # "cli_command", "file_edit", "session_create", etc.
    description: str
    command: Optional[str] = None
    session_id: Optional[int] = None
    file_path: Optional[str] = None
    message: Optional[discord.Message] = None
    status: ApprovalStatus = ApprovalStatus.PENDING


class PermissionManager:
    """
    Manages permission requests via Discord reactions.

    Before Altair executes sensitive actions, it:
    1. Posts what it wants to do
    2. Adds approve/reject reaction buttons
    3. Waits for user reaction
    4. Proceeds only if approved
    """

    APPROVE_EMOJI = "✅"
    REJECT_EMOJI = "❌"

    def __init__(
        self,
        bot: discord.Client,
        allowed_user_id: int,
        timeout: float = 300.0
    ):
        """
        Initialize permission manager.

        Args:
            bot: Discord bot client
            allowed_user_id: User ID who can approve/reject
            timeout: Timeout in seconds (default 5 minutes)
        """
        self.bot = bot
        self.allowed_user_id = allowed_user_id
        self.timeout = timeout
        self._pending_requests: dict[int, PermissionRequest] = {}  # message_id -> request

    def format_permission_message(self, request: PermissionRequest) -> str:
        """Format a permission request as a Discord message."""
        lines = ["**Permission Request**", ""]

        if request.action_type == "cli_command":
            lines.append("I want to run:")
            lines.append(f"```\n{request.command}\n```")
            if request.session_id:
                lines.append(f"in session #{request.session_id}")

        elif request.action_type == "file_edit":
            lines.append("I want to edit:")
            lines.append(f"`{request.file_path}`")
            if request.description:
                lines.append(f"\nChange: {request.description}")

        elif request.action_type == "session_create":
            lines.append("I want to create a new terminal session")
            if request.description:
                lines.append(f"\nPurpose: {request.description}")

        elif request.action_type == "start_claude":
            lines.append("I want to start Claude Code")
            if request.session_id:
                lines.append(f"in session #{request.session_id}")
            if request.command:
                lines.append(f"\nCommand: `{request.command}`")

        else:
            lines.append(f"Action: {request.action_type}")
            if request.description:
                lines.append(f"\n{request.description}")

        lines.append("")
        lines.append(f"React {self.APPROVE_EMOJI} to approve or {self.REJECT_EMOJI} to reject.")

        return "\n".join(lines)

    async def request_permission(
        self,
        channel: discord.TextChannel,
        action_type: str,
        description: str,
        command: Optional[str] = None,
        session_id: Optional[int] = None,
        file_path: Optional[str] = None,
    ) -> ApprovalStatus:
        """
        Request permission for an action.

        Posts a message asking for approval, waits for reaction.

        Args:
            channel: Discord channel to post in
            action_type: Type of action (cli_command, file_edit, etc.)
            description: Human-readable description
            command: Command to run (for cli_command)
            session_id: Session ID (for cli_command, start_claude)
            file_path: File path (for file_edit)

        Returns:
            ApprovalStatus indicating whether approved, rejected, or timeout
        """
        request = PermissionRequest(
            action_type=action_type,
            description=description,
            command=command,
            session_id=session_id,
            file_path=file_path,
        )

        # Format and send the permission request message
        content = self.format_permission_message(request)

        try:
            message = await channel.send(content)
            request.message = message

            # Add reaction buttons
            await message.add_reaction(self.APPROVE_EMOJI)
            await message.add_reaction(self.REJECT_EMOJI)

            # Store pending request
            self._pending_requests[message.id] = request

            # Wait for reaction
            status = await self._wait_for_reaction(message)

            # Update message with result
            await self._update_message_status(message, status)

            # Cleanup
            del self._pending_requests[message.id]

            return status

        except Exception as e:
            logger.error(f"Permission request failed: {e}")
            return ApprovalStatus.ERROR

    async def _wait_for_reaction(self, message: discord.Message) -> ApprovalStatus:
        """Wait for user reaction on a permission message."""

        def check(reaction: discord.Reaction, user: discord.User) -> bool:
            return (
                reaction.message.id == message.id
                and user.id == self.allowed_user_id
                and str(reaction.emoji) in [self.APPROVE_EMOJI, self.REJECT_EMOJI]
            )

        try:
            reaction, user = await self.bot.wait_for(
                "reaction_add",
                timeout=self.timeout,
                check=check
            )

            if str(reaction.emoji) == self.APPROVE_EMOJI:
                return ApprovalStatus.APPROVED
            else:
                return ApprovalStatus.REJECTED

        except asyncio.TimeoutError:
            return ApprovalStatus.TIMEOUT

    async def _update_message_status(
        self,
        message: discord.Message,
        status: ApprovalStatus
    ):
        """Update the permission message with the result status."""
        status_text = {
            ApprovalStatus.APPROVED: f"**Status: APPROVED** {self.APPROVE_EMOJI}",
            ApprovalStatus.REJECTED: f"**Status: REJECTED** {self.REJECT_EMOJI}",
            ApprovalStatus.TIMEOUT: "**Status: TIMEOUT** (no response)",
            ApprovalStatus.ERROR: "**Status: ERROR**",
        }

        try:
            # Get original content and append status
            original_content = message.content
            new_content = f"{original_content}\n\n{status_text.get(status, 'Unknown status')}"

            await message.edit(content=new_content)

        except Exception as e:
            logger.warning(f"Failed to update permission message: {e}")

    def handle_reaction(self, reaction: discord.Reaction, user: discord.User) -> Optional[ApprovalStatus]:
        """
        Handle a reaction on a permission message.

        Called by the bot's on_reaction_add handler.

        Returns:
            ApprovalStatus if this was a pending permission, None otherwise
        """
        if user.id != self.allowed_user_id:
            return None

        if reaction.message.id not in self._pending_requests:
            return None

        emoji = str(reaction.emoji)
        if emoji == self.APPROVE_EMOJI:
            return ApprovalStatus.APPROVED
        elif emoji == self.REJECT_EMOJI:
            return ApprovalStatus.REJECTED

        return None
