"""Discord UI views for department approval workflows."""

import asyncio
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)


class DepartmentApprovalView(discord.ui.View):
    """Button view for department creation/closure approval."""

    def __init__(self, allowed_user_id: int, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.allowed_user_id = allowed_user_id
        self.result: Optional[bool] = None  # True=approved, False=rejected, None=timeout
        self._event = asyncio.Event()

    async def wait_for_result(self) -> Optional[bool]:
        """Wait for user to press a button or timeout."""
        await self._event.wait()
        return self.result

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, emoji="\u2705")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.allowed_user_id:
            await interaction.response.send_message(
                "Only the bot owner can approve this.", ephemeral=True
            )
            return
        self.result = True
        self._disable_all()
        await interaction.response.edit_message(view=self)
        self._event.set()

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.red, emoji="\u274c")
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.allowed_user_id:
            await interaction.response.send_message(
                "Only the bot owner can reject this.", ephemeral=True
            )
            return
        self.result = False
        self._disable_all()
        await interaction.response.edit_message(view=self)
        self._event.set()

    async def on_timeout(self):
        self._disable_all()
        self._event.set()

    def _disable_all(self):
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True


def build_department_proposal_embed(
    project_name: str, channel_id: int, reason: str
) -> discord.Embed:
    """Build an embed for proposing a department."""
    embed = discord.Embed(
        title=f"Department Proposal: {project_name}",
        description=(
            f"Vega proposes converting the meeting room <#{channel_id}> "
            f"into a persistent **department** channel.\n\n"
            f"**Reason:** {reason}"
        ),
        color=0x5865F2,
    )
    embed.add_field(
        name="What this means",
        value=(
            "The channel will persist after plans complete and be reused "
            "for future work on this project."
        ),
        inline=False,
    )
    embed.set_footer(text="Awaiting approval...")
    return embed


def build_department_close_embed(
    project_name: str, channel_id: int, reason: str
) -> discord.Embed:
    """Build an embed for proposing department closure."""
    embed = discord.Embed(
        title=f"Close Department: {project_name}",
        description=(
            f"Vega proposes closing the department <#{channel_id}>.\n\n"
            f"**Reason:** {reason}"
        ),
        color=0xE67E22,
    )
    embed.set_footer(text="Awaiting approval...")
    return embed
