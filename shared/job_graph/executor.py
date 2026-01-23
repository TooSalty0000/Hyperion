"""Job Graph Executor - manages active graphs and dispatches work."""

import logging
from typing import Optional

import discord
from discord.ext import commands

from .models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus

logger = logging.getLogger(__name__)

# Embed colors
COLOR_PLAN = 0x5865F2      # Blurple - plan created
COLOR_DISPATCH = 0xFEE75C  # Yellow - node dispatched
COLOR_COMPLETE = 0x57F287  # Green - node/graph completed
COLOR_FAILED = 0xED4245    # Red - node/graph failed
COLOR_TIMEOUT = 0xE67E22   # Orange - timeout
COLOR_STATUS = 0x99AAB5    # Grey - status update


class JobGraphExecutor:
    """
    Manages active job graphs and dispatches work to agents.

    Responsibilities:
    - Track all active graphs across channels
    - Create Discord threads for planning visibility
    - Dispatch READY nodes by sending @mentions
    - Handle timeouts on running nodes
    - Provide graph state context for Vega's LLM
    """

    def __init__(
        self,
        bot: commands.Bot,
        agent_registry: dict[str, int],
        max_timeout: int = 600,
    ):
        self.bot = bot
        self.agent_registry = agent_registry
        self.max_timeout = max_timeout

        # Active graphs: channel_id → list of graphs (multiple can be active)
        self._graphs: dict[int, list[JobGraph]] = {}

        # Thread ID → graph mapping for quick lookup
        self._thread_to_graph: dict[int, JobGraph] = {}

    def get_active_graphs(self, channel_id: int) -> list[JobGraph]:
        """Get all active graphs for a channel."""
        graphs = self._graphs.get(channel_id, [])
        return [g for g in graphs if g.status == GraphStatus.ACTIVE]

    def get_all_active_graphs(self) -> list[JobGraph]:
        """Get all active graphs across all channels."""
        result = []
        for graphs in self._graphs.values():
            result.extend(g for g in graphs if g.status == GraphStatus.ACTIVE)
        return result

    def get_graph_by_thread(self, thread_id: int) -> Optional[JobGraph]:
        """Get a graph by its thread ID."""
        return self._thread_to_graph.get(thread_id)

    async def create_graph(
        self,
        goal: str,
        trigger_message: discord.Message,
    ) -> JobGraph:
        """
        Create a new job graph and its Discord thread.

        Args:
            goal: The user's goal/request
            trigger_message: The Discord message that triggered this graph

        Returns:
            The newly created JobGraph
        """
        graph = JobGraph(
            goal=goal,
            trigger_message_id=trigger_message.id,
            channel_id=trigger_message.channel.id,
            max_timeout=self.max_timeout,
        )

        # Create a Discord thread under the user's message for planning visibility
        try:
            thread = await trigger_message.create_thread(
                name=f"Plan: {goal[:80]}",
                auto_archive_duration=60,
            )
            graph.thread_id = thread.id
            self._thread_to_graph[thread.id] = graph
            logger.info(f"[Executor] Created thread {thread.id} for graph {graph.id}")
        except discord.HTTPException as e:
            logger.warning(f"[Executor] Failed to create thread: {e}")

        # Track the graph
        channel_id = trigger_message.channel.id
        if channel_id not in self._graphs:
            self._graphs[channel_id] = []
        self._graphs[channel_id].append(graph)

        return graph

    async def dispatch_ready_nodes(self, graph: JobGraph) -> list[JobNode]:
        """
        Dispatch all READY nodes in a graph.

        For DISPATCH nodes: sends @mentions to the target agent in the main channel.
        For THINK nodes: marks as running (Vega handles internally).
        For RESPOND nodes: marks as running (Vega handles internally).

        Returns:
            List of nodes that were dispatched
        """
        dispatched = []
        ready_nodes = graph.get_ready_nodes()

        for node in ready_nodes:
            if node.type == NodeType.DISPATCH:
                success = await self._dispatch_to_agent(graph, node)
                if success:
                    dispatched.append(node)
            elif node.type in (NodeType.THINK, NodeType.RESPOND):
                # These are handled by Vega's LLM directly
                graph.mark_running(node.id)
                dispatched.append(node)

        # Update thread with dispatch status
        if dispatched and graph.thread_id:
            await self._post_status_embed(graph)

        return dispatched

    async def _dispatch_to_agent(self, graph: JobGraph, node: JobNode) -> bool:
        """Send an @mention to an agent for a DISPATCH node."""
        if not node.agent:
            logger.error(f"[Executor] DISPATCH node {node.id} has no agent assigned")
            graph.mark_failed(node.id, "No agent assigned")
            return False

        agent_id = self.agent_registry.get(node.agent)
        if not agent_id:
            logger.error(f"[Executor] Unknown agent: {node.agent}")
            graph.mark_failed(node.id, f"Unknown agent: {node.agent}")
            return False

        channel = self.bot.get_channel(graph.channel_id)
        if not channel:
            logger.error(f"[Executor] Channel {graph.channel_id} not found")
            graph.mark_failed(node.id, "Channel not found")
            return False

        # Build the mention message
        mention = f"<@{agent_id}>"
        message_content = f"{mention} {node.description}"

        try:
            sent_msg = await channel.send(message_content)
            graph.mark_running(node.id, message_id=sent_msg.id)
            logger.info(
                f"[Executor] Dispatched node {node.id} to {node.agent}: "
                f"{node.description[:50]}..."
            )

            # Post dispatch embed to thread
            await self._post_dispatch_embed(graph, node)
            return True
        except discord.HTTPException as e:
            logger.error(f"[Executor] Failed to dispatch node {node.id}: {e}")
            graph.mark_failed(node.id, f"Discord send failed: {e}")
            return False

    def check_timeouts(self) -> list[tuple[JobGraph, JobNode]]:
        """
        Check all active graphs for timed-out nodes.

        Returns:
            List of (graph, node) tuples for nodes that timed out
        """
        timed_out = []
        for graphs in self._graphs.values():
            for graph in graphs:
                if graph.status != GraphStatus.ACTIVE:
                    continue
                for node in graph.get_timed_out_nodes():
                    graph.mark_failed(node.id, f"Timed out after {node.timeout}s")
                    timed_out.append((graph, node))
        return timed_out

    def find_node_for_agent_response(
        self,
        agent_name: str,
        channel_id: int,
    ) -> Optional[tuple[JobGraph, JobNode]]:
        """
        Find the running DISPATCH node that an agent response likely corresponds to.

        Matches by: agent name + channel + RUNNING status.
        If multiple match, returns the oldest (first dispatched).
        """
        candidates = []

        for graph in self.get_active_graphs(channel_id):
            for node in graph.get_running_nodes():
                if node.type == NodeType.DISPATCH and node.agent == agent_name:
                    candidates.append((graph, node))

        if not candidates:
            return None

        # Return the oldest dispatched node (FIFO)
        candidates.sort(key=lambda x: x[1].dispatched_at or x[1].created_at)
        return candidates[0]

    def get_all_graphs_context(self) -> str:
        """
        Get context string for all active graphs.

        This is included in Vega's system prompt so she's fully aware
        of everything happening.
        """
        active = self.get_all_active_graphs()
        if not active:
            return ""

        lines = ["ACTIVE JOB GRAPHS:"]
        for graph in active:
            lines.append(graph.get_context_for_llm())
            lines.append("")

        return "\n".join(lines)

    async def complete_graph(self, graph: JobGraph) -> None:
        """Mark a graph as completed and delete the planning thread."""
        graph.status = GraphStatus.COMPLETED

        # Post completion embed then delete the thread
        await self._post_completion_embed(graph, success=True)
        await self._delete_thread(graph)

        # Clean up thread mapping
        if graph.thread_id and graph.thread_id in self._thread_to_graph:
            del self._thread_to_graph[graph.thread_id]

    async def fail_graph(self, graph: JobGraph) -> None:
        """Mark a graph as failed and delete the planning thread."""
        graph.status = GraphStatus.FAILED

        # Post failure embed then delete the thread
        await self._post_completion_embed(graph, success=False)
        await self._delete_thread(graph)

        # Clean up thread mapping
        if graph.thread_id and graph.thread_id in self._thread_to_graph:
            del self._thread_to_graph[graph.thread_id]

    # --------------------------------------------------
    # EMBED HELPERS
    # --------------------------------------------------

    async def _post_plan_embed(self, graph: JobGraph) -> None:
        """Post the initial plan creation embed to the thread."""
        if not graph.thread_id:
            return

        embed = discord.Embed(
            title="Plan Created",
            description=f"**Goal:** {graph.goal}",
            color=COLOR_PLAN,
        )

        # List nodes
        node_lines = []
        for node in graph.nodes.values():
            node_lines.append(node.to_summary())

        if node_lines:
            embed.add_field(
                name=f"Tasks ({len(node_lines)})",
                value="\n".join(node_lines),
                inline=False,
            )

        embed.set_footer(text=f"Graph: {graph.id}")

        await self._send_to_thread(graph, embed=embed)

    async def _post_status_embed(self, graph: JobGraph) -> None:
        """Post a status update embed."""
        if not graph.thread_id:
            return

        embed = discord.Embed(
            color=COLOR_STATUS,
        )

        node_lines = []
        for node in graph.nodes.values():
            node_lines.append(node.to_summary())

        if node_lines:
            embed.description = "\n".join(node_lines)

        # Status counts in footer
        counts = {}
        for node in graph.nodes.values():
            counts[node.status.value] = counts.get(node.status.value, 0) + 1
        status_str = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        embed.set_footer(text=status_str)

        await self._send_to_thread(graph, embed=embed)

    async def _post_dispatch_embed(self, graph: JobGraph, node: JobNode) -> None:
        """Post a dispatch notification embed."""
        if not graph.thread_id:
            return

        embed = discord.Embed(
            description=f"**{node.agent}** ← `{node.id}`: {node.description}",
            color=COLOR_DISPATCH,
        )
        embed.set_footer(text=f"Timeout: {node.timeout}s")

        await self._send_to_thread(graph, embed=embed)

    async def _post_node_update_embed(
        self, graph: JobGraph, node: JobNode, status: str, detail: str = ""
    ) -> None:
        """Post a node status change embed."""
        if not graph.thread_id:
            return

        if status == "completed":
            color = COLOR_COMPLETE
        elif status == "failed":
            color = COLOR_FAILED
        else:
            color = COLOR_STATUS

        desc = f"`{node.id}` → **{status}**"
        if detail:
            desc += f"\n{detail[:200]}"

        embed = discord.Embed(description=desc, color=color)
        await self._send_to_thread(graph, embed=embed)

    async def _post_completion_embed(self, graph: JobGraph, success: bool) -> None:
        """Post the final graph completion embed."""
        if not graph.thread_id:
            return

        if success:
            embed = discord.Embed(
                title="Plan Complete",
                description=f"Goal achieved: {graph.goal}",
                color=COLOR_COMPLETE,
            )
        else:
            embed = discord.Embed(
                title="Plan Failed",
                description=f"Goal: {graph.goal}",
                color=COLOR_FAILED,
            )

        # Summarize node outcomes
        completed = sum(1 for n in graph.nodes.values() if n.status == NodeStatus.COMPLETED)
        failed = sum(1 for n in graph.nodes.values() if n.status == NodeStatus.FAILED)
        cancelled = sum(1 for n in graph.nodes.values() if n.status == NodeStatus.CANCELLED)

        summary = f"{completed} completed"
        if failed:
            summary += f", {failed} failed"
        if cancelled:
            summary += f", {cancelled} cancelled"
        embed.set_footer(text=summary)

        await self._send_to_thread(graph, embed=embed)

    async def _post_timeout_embed(self, graph: JobGraph, node: JobNode) -> None:
        """Post a timeout notification embed."""
        if not graph.thread_id:
            return

        embed = discord.Embed(
            description=f"`{node.id}` ({node.agent}) timed out after **{node.timeout}s**",
            color=COLOR_TIMEOUT,
        )
        await self._send_to_thread(graph, embed=embed)

    # --------------------------------------------------
    # THREAD MANAGEMENT
    # --------------------------------------------------

    async def _send_to_thread(
        self, graph: JobGraph, content: str = None, embed: discord.Embed = None
    ) -> None:
        """Send a message or embed to a graph's planning thread."""
        if not graph.thread_id:
            return

        try:
            thread = self.bot.get_channel(graph.thread_id)
            if thread:
                await thread.send(content=content, embed=embed)
        except discord.HTTPException as e:
            logger.warning(f"[Executor] Failed to send to thread: {e}")

    async def _delete_thread(self, graph: JobGraph) -> None:
        """Delete a graph's planning thread."""
        if not graph.thread_id:
            return

        try:
            thread = self.bot.get_channel(graph.thread_id)
            if thread:
                await thread.delete()
                logger.info(f"[Executor] Deleted thread {graph.thread_id} for graph {graph.id}")
        except discord.HTTPException as e:
            logger.warning(f"[Executor] Failed to delete thread: {e}")

    async def post_to_thread(self, graph: JobGraph, message: str) -> None:
        """Post a plain text message to a graph's planning thread (legacy compat)."""
        if not graph.thread_id:
            return

        try:
            thread = self.bot.get_channel(graph.thread_id)
            if thread:
                if len(message) > 1900:
                    message = message[:1900] + "..."
                await thread.send(message)
        except discord.HTTPException as e:
            logger.warning(f"[Executor] Failed to post to thread: {e}")

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def cleanup_completed(self, max_age_minutes: int = 30) -> int:
        """
        Remove completed/failed graphs older than max_age_minutes.

        Returns number of graphs cleaned up.
        """
        from datetime import timedelta

        cutoff = self._now() - timedelta(minutes=max_age_minutes)
        cleaned = 0

        for channel_id in list(self._graphs.keys()):
            remaining = []
            for graph in self._graphs[channel_id]:
                if graph.status != GraphStatus.ACTIVE and graph.created_at < cutoff:
                    if graph.thread_id and graph.thread_id in self._thread_to_graph:
                        del self._thread_to_graph[graph.thread_id]
                    cleaned += 1
                else:
                    remaining.append(graph)
            self._graphs[channel_id] = remaining

        return cleaned

    @staticmethod
    def _now():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
