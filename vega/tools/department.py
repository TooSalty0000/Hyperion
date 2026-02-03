"""Department tools for persistent project channels."""

import asyncio
import logging
from typing import List

import discord

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class ProposeDepartmentTool(Tool):
    """
    Propose converting a meeting room into a persistent department channel.

    Sends an approval request with buttons to the user. Non-blocking:
    fires a background task for the approval flow.
    """

    @property
    def name(self) -> str:
        return "propose_department"

    @property
    def description(self) -> str:
        return (
            "Propose converting the current plan's meeting room into a persistent "
            "department channel tied to a project. Sends an approval request with "
            "buttons to the user. The channel will persist across plans if approved."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="project_name",
                type="string",
                description="Name of the project to create a department for",
                required=True,
            ),
            ToolParameter(
                name="graph_id",
                type="string",
                description="ID of the graph whose meeting room to promote",
                required=True,
            ),
            ToolParameter(
                name="reason",
                type="string",
                description="Why this project needs a persistent department",
                required=True,
            ),
        ]

    async def execute(
        self, context: ToolContext, project_name: str, graph_id: str, reason: str
    ) -> str:
        executor = context.job_executor
        if not executor:
            return "Error: Job executor not available."

        # Find the graph
        graph = None
        for g in executor.get_all_active_graphs():
            if g.id == graph_id:
                graph = g
                break

        if not graph:
            return f"Error: Graph '{graph_id}' not found or not active."

        if not graph.project_channel_id:
            return "Error: This graph has no meeting room to promote."

        # Check if already a department
        if executor.is_department_channel(graph.project_channel_id):
            return f"This channel is already a department."

        # Find the main channel to send the approval embed
        trigger_msg = context.trigger_message
        if trigger_msg:
            main_channel = trigger_msg.channel
        else:
            main_channel_obj = executor.bot.get_channel(graph.channel_id)
            if not main_channel_obj:
                return "Error: Cannot find main channel for approval."
            main_channel = main_channel_obj

        # Build and send approval embed with buttons
        from vega.views.department import (
            DepartmentApprovalView,
            build_department_proposal_embed,
        )
        from vega.config import Config

        allowed_user_id = Config.ALLOWED_USER_ID
        if not allowed_user_id:
            return "Error: ALLOWED_USER_ID not configured."

        embed = build_department_proposal_embed(
            project_name, graph.project_channel_id, reason
        )
        view = DepartmentApprovalView(allowed_user_id=allowed_user_id)

        try:
            approval_msg = await main_channel.send(embed=embed, view=view)
        except discord.HTTPException as e:
            return f"Error: Failed to send approval embed: {e}"

        # Fire background task for approval handling (non-blocking)
        asyncio.create_task(
            self._handle_approval(
                executor=executor,
                graph=graph,
                project_name=project_name,
                view=view,
                approval_msg=approval_msg,
                project_manager=context.project_manager,
            )
        )

        return (
            f"Department proposal sent for '{project_name}'. "
            f"Awaiting user approval via buttons."
        )

    async def _handle_approval(
        self, executor, graph, project_name, view, approval_msg, project_manager
    ):
        """Background handler for department approval."""
        try:
            result = await view.wait_for_result()

            if result is True:
                # Approved - promote channel
                channel_id = graph.project_channel_id
                channel_manager = executor.channel_manager

                if channel_manager:
                    success = await channel_manager.promote_to_department(
                        channel_id, project_name
                    )
                    if not success:
                        embed = approval_msg.embeds[0] if approval_msg.embeds else None
                        if embed:
                            embed.set_footer(text="Failed to rename channel")
                            try:
                                await approval_msg.edit(embed=embed)
                            except discord.HTTPException:
                                pass
                        return

                # Register with executor
                executor.register_department(channel_id, project_name)
                graph.project_name = project_name

                # Update project in Redis
                if project_manager:
                    try:
                        project = await project_manager.get(project_name)
                        if project:
                            await project_manager.update(
                                project_name, channel_id=channel_id
                            )
                        else:
                            # Create project if it doesn't exist
                            await project_manager.create(
                                name=project_name,
                                path="",  # Path can be set later
                                settings={"department_channel_id": channel_id},
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to update project in Redis: {e}"
                        )

                # Update embed footer
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Approved")
                    embed.color = 0x57F287  # Green
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass

                # Post notice in department channel
                dept_channel = executor.bot.get_channel(channel_id)
                if dept_channel:
                    try:
                        notice_embed = discord.Embed(
                            title=f"Department: {project_name}",
                            description=(
                                "This channel has been promoted to a persistent department. "
                                "It will be reused for future work on this project."
                            ),
                            color=0x57F287,
                        )
                        await dept_channel.send(embed=notice_embed)
                    except discord.HTTPException:
                        pass

                logger.info(
                    f"Department approved: {project_name} -> channel {channel_id}"
                )

            elif result is False:
                # Rejected
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Rejected")
                    embed.color = 0xED4245  # Red
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass
                logger.info(f"Department rejected: {project_name}")

            else:
                # Timeout
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Timed out")
                    embed.color = 0x99AAB5  # Grey
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass
                logger.info(f"Department proposal timed out: {project_name}")

        except Exception as e:
            logger.error(f"Department approval handler error: {e}", exc_info=True)


class CloseDepartmentTool(Tool):
    """
    Propose closing a persistent department channel.

    Sends an approval request with buttons to the user.
    """

    @property
    def name(self) -> str:
        return "close_department"

    @property
    def description(self) -> str:
        return (
            "Propose closing a persistent department channel. Sends an approval "
            "request with buttons to the user. If approved, the channel is deleted "
            "and the project's channel_id is cleared."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="project_name",
                type="string",
                description="Name of the project whose department to close",
                required=True,
            ),
            ToolParameter(
                name="reason",
                type="string",
                description="Why this department should be closed",
                required=True,
            ),
        ]

    async def execute(
        self, context: ToolContext, project_name: str, reason: str
    ) -> str:
        executor = context.job_executor
        if not executor:
            return "Error: Job executor not available."

        project_manager = context.project_manager
        if not project_manager:
            return "Error: Project manager not available."

        # Look up the project
        project = await project_manager.get(project_name)
        if not project:
            return f"Error: Project '{project_name}' not found."

        if not project.channel_id:
            return f"Error: Project '{project_name}' has no department channel."

        channel_id = project.channel_id

        if not executor.is_department_channel(channel_id):
            return f"Error: Channel is not a registered department."

        # Find the main channel to send the approval embed
        trigger_msg = context.trigger_message
        if trigger_msg:
            main_channel = trigger_msg.channel
        else:
            # Try to find a main channel from active graphs
            main_channel = None
            for graphs in executor._graphs.values():
                for g in graphs:
                    if g.channel_id:
                        main_channel = executor.bot.get_channel(g.channel_id)
                        if main_channel:
                            break
                if main_channel:
                    break

        if not main_channel:
            return "Error: Cannot find main channel for approval."

        from vega.views.department import (
            DepartmentApprovalView,
            build_department_close_embed,
        )
        from vega.config import Config

        allowed_user_id = Config.ALLOWED_USER_ID
        if not allowed_user_id:
            return "Error: ALLOWED_USER_ID not configured."

        embed = build_department_close_embed(project_name, channel_id, reason)
        view = DepartmentApprovalView(allowed_user_id=allowed_user_id)

        try:
            approval_msg = await main_channel.send(embed=embed, view=view)
        except discord.HTTPException as e:
            return f"Error: Failed to send close approval embed: {e}"

        # Fire background task
        asyncio.create_task(
            self._handle_close_approval(
                executor=executor,
                project_name=project_name,
                channel_id=channel_id,
                view=view,
                approval_msg=approval_msg,
                project_manager=project_manager,
            )
        )

        return (
            f"Department close proposal sent for '{project_name}'. "
            f"Awaiting user approval via buttons."
        )

    async def _handle_close_approval(
        self, executor, project_name, channel_id, view, approval_msg, project_manager
    ):
        """Background handler for department close approval."""
        try:
            result = await view.wait_for_result()

            if result is True:
                # Approved - delete channel
                if executor.channel_manager:
                    await executor.channel_manager.delete_project_channel(channel_id)

                executor.unregister_department(channel_id)

                # Clear channel_id from project
                try:
                    await project_manager.update(project_name, channel_id=None)
                except Exception as e:
                    logger.error(f"Failed to clear project channel_id: {e}")

                # Clean up graph list for this channel
                if channel_id in executor._project_channel_to_graphs:
                    del executor._project_channel_to_graphs[channel_id]

                # Update embed
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Closed")
                    embed.color = 0x57F287
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass

                logger.info(f"Department closed: {project_name}")

            elif result is False:
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Rejected")
                    embed.color = 0xED4245
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass
                logger.info(f"Department close rejected: {project_name}")

            else:
                embed = approval_msg.embeds[0] if approval_msg.embeds else None
                if embed:
                    embed.set_footer(text="Timed out")
                    embed.color = 0x99AAB5
                    try:
                        await approval_msg.edit(embed=embed, view=view)
                    except discord.HTTPException:
                        pass
                logger.info(f"Department close timed out: {project_name}")

        except Exception as e:
            logger.error(f"Department close handler error: {e}", exc_info=True)


def get_department_tools() -> list[Tool]:
    """Get all department management tools."""
    return [
        ProposeDepartmentTool(),
        CloseDepartmentTool(),
    ]
