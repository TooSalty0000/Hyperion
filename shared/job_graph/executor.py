"""Job Graph Executor - manages active graphs and dispatches work."""

import asyncio
import logging
from typing import Optional, Callable, Awaitable, TYPE_CHECKING

import discord
from discord.ext import commands

from .models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus
from shared.agent_messaging import format_node_marker

if TYPE_CHECKING:
    from shared.channel.manager import ChannelManager

logger = logging.getLogger(__name__)

# Embed color for meeting room intro
COLOR_PLAN = 0x5865F2      # Blurple


class JobGraphExecutor:
    """
    Manages active job graphs and dispatches work to agents.

    Responsibilities:
    - Track all active graphs across channels
    - Dispatch READY nodes by sending @mentions
    - Handle timeouts on running nodes
    - Provide graph state context for Vega's LLM
    """

    def __init__(
        self,
        bot: commands.Bot,
        agent_registry: dict[str, int],
        max_timeout: int = 600,
        on_timeout_callback: Optional[Callable[[JobGraph, JobNode], Awaitable[None]]] = None,
    ):
        self.bot = bot
        self.agent_registry = agent_registry
        self.max_timeout = max_timeout
        self._on_timeout_callback = on_timeout_callback

        # Active graphs: channel_id → list of graphs (multiple can be active)
        self._graphs: dict[int, list[JobGraph]] = {}

        # Project channel (meeting room / department) → list of graphs (1:many for departments)
        self._project_channel_to_graphs: dict[int, list[JobGraph]] = {}

        # Department channel tracking (persistent channels that survive graph completion)
        self._department_channel_ids: set[int] = set()
        self._department_info: dict[int, str] = {}  # channel_id -> project_name

        # Per-node timeout tasks: "graph_id:node_id" → asyncio.Task
        self._timeout_tasks: dict[str, asyncio.Task] = {}

        # Channel manager for creating/deleting meeting rooms (set by VegaCore)
        self.channel_manager: Optional["ChannelManager"] = None

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

    async def create_graph(
        self,
        goal: str,
        trigger_message: discord.Message,
    ) -> JobGraph:
        """
        Create a new job graph.

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
        Sequential-per-agent: only one DISPATCH node per agent may be RUNNING
        at a time (across all active graphs).  Additional READY nodes for a
        busy agent are deferred until the running node completes or fails.
        For THINK nodes: auto-completes (reasoning already done during planning).
        For RESPOND nodes: marks as running (Vega handles via respond_to_user).

        Returns:
            List of nodes that were dispatched
        """
        dispatched = []
        ready_nodes = graph.get_ready_nodes()

        # Sequential-per-agent: track which agents already have running work
        busy_agents = self._get_running_dispatch_agents()

        for node in ready_nodes:
            if node.type == NodeType.DISPATCH:
                if node.agent in busy_agents:
                    logger.debug(
                        f"[Executor] Deferring node '{node.id}' — "
                        f"agent '{node.agent}' already has a running dispatch node"
                    )
                    continue
                success = await self._dispatch_to_agent(graph, node)
                if success:
                    dispatched.append(node)
                    busy_agents.add(node.agent)
            elif node.type == NodeType.THINK:
                # Think = Vega's internal reasoning, already done when plan was created.
                # Auto-complete to unblock dependent dispatch nodes.
                graph.mark_completed(node.id, "Internal reasoning completed")
                dispatched.append(node)
            elif node.type == NodeType.RESPOND:
                graph.mark_running(node.id)
                dispatched.append(node)

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

        # Use meeting room if available, fall back to main channel
        dispatch_channel_id = graph.project_channel_id or graph.channel_id
        channel = self.bot.get_channel(dispatch_channel_id)
        if not channel:
            logger.error(f"[Executor] Channel {dispatch_channel_id} not found")
            graph.mark_failed(node.id, "Channel not found")
            return False

        # Build the mention message with node marker for response matching
        mention = f"<@{agent_id}>"
        marker = format_node_marker(graph.id, node.id)
        message_content = f"{mention} {marker} {node.description}"

        try:
            sent_msg = await channel.send(message_content, silent=True)
            graph.mark_running(node.id, message_id=sent_msg.id)
            logger.info(
                f"[Executor] Dispatched node {node.id} to {node.agent}: "
                f"{node.description[:50]}..."
            )

            # Schedule per-node timeout
            self._schedule_timeout(graph, node)
            return True
        except discord.HTTPException as e:
            logger.error(f"[Executor] Failed to dispatch node {node.id}: {e}")
            graph.mark_failed(node.id, f"Discord send failed: {e}")
            return False

    def check_timeouts(self) -> list[tuple[JobGraph, JobNode]]:
        """
        Check all active graphs for timed-out nodes.

        DEPRECATED: Retained for backward compatibility. Per-node asyncio timeout
        tasks now handle timeouts directly via _schedule_timeout().

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

    # --------------------------------------------------
    # PER-NODE TIMEOUT TASKS
    # --------------------------------------------------

    def _schedule_timeout(self, graph: JobGraph, node: JobNode) -> None:
        """Schedule an asyncio task that fires after node.timeout seconds."""
        key = f"{graph.id}:{node.id}"

        async def _timeout_coro():
            try:
                await asyncio.sleep(node.timeout)
                # Re-check: node may have been completed/cancelled while sleeping
                if node.status != NodeStatus.RUNNING:
                    return
                graph.mark_failed(node.id, f"Timed out after {node.timeout}s")
                logger.warning(
                    f"[Executor] Node '{node.id}' timed out in graph '{graph.id}' "
                    f"(agent={node.agent}, timeout={node.timeout}s)"
                )
                # Fire callback (Vega re-evaluation)
                if self._on_timeout_callback:
                    try:
                        await self._on_timeout_callback(graph, node)
                    except Exception as cb_err:
                        logger.error(f"[Executor] Timeout callback error: {cb_err}")
                # Sequential-per-agent: agent is now free, dispatch next queued node
                if node.agent:
                    try:
                        await self.dispatch_next_for_agent(node.agent)
                    except Exception as dispatch_err:
                        logger.error(
                            f"[Executor] dispatch_next_for_agent error: {dispatch_err}"
                        )
            except asyncio.CancelledError:
                pass  # Normal cancellation when agent responds in time
            finally:
                self._timeout_tasks.pop(key, None)

        task = asyncio.create_task(_timeout_coro())
        self._timeout_tasks[key] = task
        logger.debug(f"[Executor] Scheduled {node.timeout}s timeout for {key}")

    def cancel_timeout(self, graph_id: str, node_id: str) -> None:
        """Cancel a specific node's timeout task."""
        key = f"{graph_id}:{node_id}"
        task = self._timeout_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()
            logger.debug(f"[Executor] Cancelled timeout for {key}")

    def cancel_all_timeouts(self, graph_id: str) -> None:
        """Cancel all timeout tasks for a graph."""
        prefix = f"{graph_id}:"
        keys_to_remove = [k for k in self._timeout_tasks if k.startswith(prefix)]
        for key in keys_to_remove:
            task = self._timeout_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()
        if keys_to_remove:
            logger.debug(f"[Executor] Cancelled {len(keys_to_remove)} timeouts for graph {graph_id}")

    # --------------------------------------------------
    # SEQUENTIAL-PER-AGENT DISPATCH
    # --------------------------------------------------

    def _get_running_dispatch_agents(self) -> set[str]:
        """Get agent names that currently have RUNNING dispatch nodes across all active graphs."""
        running = set()
        for graphs in self._graphs.values():
            for graph in graphs:
                if graph.status != GraphStatus.ACTIVE:
                    continue
                for node in graph.nodes.values():
                    if (node.type == NodeType.DISPATCH
                            and node.status == NodeStatus.RUNNING
                            and node.agent):
                        running.add(node.agent)
        return running

    def _agent_has_running_dispatch(self, agent_name: str) -> bool:
        """Check if a specific agent has any RUNNING dispatch node."""
        for graphs in self._graphs.values():
            for graph in graphs:
                if graph.status != GraphStatus.ACTIVE:
                    continue
                for node in graph.nodes.values():
                    if (node.type == NodeType.DISPATCH
                            and node.status == NodeStatus.RUNNING
                            and node.agent == agent_name):
                        return True
        return False

    async def dispatch_next_for_agent(self, agent_name: str) -> list[JobNode]:
        """
        Dispatch the next READY node for an agent across all active graphs.

        Called after an agent's dispatch node completes/fails to unblock
        queued work in other graphs.  Respects sequential-per-agent: only
        dispatches if the agent has no other RUNNING dispatch node.

        Returns:
            List containing the single dispatched node, or empty list.
        """
        if not agent_name or self._agent_has_running_dispatch(agent_name):
            return []

        for graphs in self._graphs.values():
            for graph in graphs:
                if graph.status != GraphStatus.ACTIVE:
                    continue
                for node in graph.get_ready_nodes():
                    if node.type == NodeType.DISPATCH and node.agent == agent_name:
                        success = await self._dispatch_to_agent(graph, node)
                        if success:
                            return [node]
        return []

    # --------------------------------------------------
    # MARKER-BASED NODE LOOKUP
    # --------------------------------------------------

    def find_node_by_marker(
        self,
        graph_id: str,
        node_id: str,
        channel_id: int,
    ) -> Optional[tuple[JobGraph, JobNode]]:
        """
        O(1) lookup of a running DISPATCH node by graph_id + node_id.

        Falls through if the node is not RUNNING (already completed/failed).
        """
        # Search graphs in the given channel and project channels
        for graphs in self._graphs.values():
            for graph in graphs:
                if graph.id != graph_id:
                    continue
                if graph.status != GraphStatus.ACTIVE:
                    continue
                node = graph.nodes.get(node_id)
                if node and node.status == NodeStatus.RUNNING and node.type == NodeType.DISPATCH:
                    return (graph, node)
        return None

    def find_node_for_agent_response(
        self,
        agent_name: str,
        channel_id: int,
    ) -> Optional[tuple[JobGraph, JobNode]]:
        """
        Find the running DISPATCH node that an agent response likely corresponds to.

        Matches by: agent name + channel + RUNNING status.
        Checks project channels (meeting rooms) first, then main channel index.
        If multiple match, returns the oldest (first dispatched).
        """
        candidates = []

        # Check project channel index first (meeting room / department responses)
        for project_graph in self._project_channel_to_graphs.get(channel_id, []):
            if project_graph.status == GraphStatus.ACTIVE:
                for node in project_graph.get_running_nodes():
                    if node.type == NodeType.DISPATCH and node.agent == agent_name:
                        candidates.append((project_graph, node))

        # Then check main channel index (no meeting room / backward compat)
        if not candidates:
            for graph in self.get_active_graphs(channel_id):
                for node in graph.get_running_nodes():
                    if node.type == NodeType.DISPATCH and node.agent == agent_name:
                        candidates.append((graph, node))

        if not candidates:
            return None

        # Return the oldest dispatched node (FIFO)
        candidates.sort(key=lambda x: x[1].dispatched_at or x[1].created_at)
        return candidates[0]

    def find_terminal_node_for_agent(
        self,
        agent_name: str,
        channel_id: int,
    ) -> Optional[tuple[JobGraph, JobNode, str]]:
        """
        Search ALL graphs (including completed/failed) for terminal nodes matching an agent.

        Used to diagnose late agent responses: was this a timed-out node the agent
        finally responded to, or a completely unsolicited message?

        Returns:
            Tuple of (graph, node, reason) if found, None otherwise.
            reason is one of: "timed_out", "completed", "failed", "cancelled"
        """
        for graphs in self._graphs.values():
            for graph in graphs:
                # Match on main channel OR project channel
                if graph.channel_id != channel_id and graph.project_channel_id != channel_id:
                    continue
                for node in graph.nodes.values():
                    if node.type != NodeType.DISPATCH or node.agent != agent_name:
                        continue
                    if node.status == NodeStatus.FAILED:
                        reason = "timed_out" if "Timed out" in (node.result or "") else "failed"
                        return (graph, node, reason)
                    if node.status == NodeStatus.COMPLETED:
                        return (graph, node, "completed")
                    if node.status == NodeStatus.CANCELLED:
                        return (graph, node, "cancelled")
        return None

    def get_channel_graphs_context(self, channel_id: int) -> str:
        """
        Get context string for active graphs in a specific channel.

        This helps Vega decide whether to extend an existing graph or create new.
        Includes agent availability so Vega can plan around busy agents.
        """
        active = self.get_active_graphs(channel_id)
        if not active:
            return "ACTIVE JOB GRAPHS: None in this channel. You may create a new plan."

        lines = ["ACTIVE JOB GRAPHS (this channel):"]
        lines.append(">>> IMPORTANT: Use `add_nodes` to extend these graphs. Do NOT create a new plan unless the task is COMPLETELY unrelated.")
        lines.append(">>> For simple follow-ups, just call `respond_to_user` with the graph_id.")
        lines.append("")

        for graph in active:
            lines.append(graph.get_context_for_llm())
            lines.append("")

        # Show agent availability so Vega can plan around busy agents
        busy_agents = self._get_running_dispatch_agents()
        if busy_agents:
            lines.append(f"BUSY AGENTS: {', '.join(sorted(busy_agents))} (dispatches to these agents will be queued)")
        else:
            lines.append("ALL AGENTS AVAILABLE")

        return "\n".join(lines)

    def get_all_graphs_context(self) -> str:
        """
        Get context string for all active graphs across all channels.

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
        """Mark a graph as completed. Delete meeting room (but not departments)."""
        graph.status = GraphStatus.COMPLETED

        # Cancel any remaining timeout tasks for this graph
        self.cancel_all_timeouts(graph.id)

        # Only delete channel if it's NOT a department
        if graph.project_channel_id and graph.project_channel_id not in self._department_channel_ids:
            await self._delete_project_channel(graph)
        elif graph.project_channel_id and graph.project_channel_id in self._department_channel_ids:
            # Remove graph from the channel's graph list but keep the channel
            self._remove_graph_from_channel(graph)

    async def fail_graph(self, graph: JobGraph) -> None:
        """Mark a graph as failed. Delete meeting room (but not departments)."""
        graph.status = GraphStatus.FAILED

        # Cancel any remaining timeout tasks for this graph
        self.cancel_all_timeouts(graph.id)

        # Only delete channel if it's NOT a department
        if graph.project_channel_id and graph.project_channel_id not in self._department_channel_ids:
            await self._delete_project_channel(graph)
        elif graph.project_channel_id and graph.project_channel_id in self._department_channel_ids:
            self._remove_graph_from_channel(graph)

    def _remove_graph_from_channel(self, graph: JobGraph) -> None:
        """Remove a graph from a project channel's graph list."""
        if not graph.project_channel_id:
            return
        graphs = self._project_channel_to_graphs.get(graph.project_channel_id, [])
        self._project_channel_to_graphs[graph.project_channel_id] = [
            g for g in graphs if g.id != graph.id
        ]

    # --------------------------------------------------
    # PROJECT CHANNELS (Meeting Rooms)
    # --------------------------------------------------

    async def create_project_channel(self, graph: JobGraph) -> Optional[int]:
        """
        Create a meeting room channel for a graph.

        Returns the channel ID, or None if creation failed (dispatches
        will fall back to the main channel).
        """
        if graph.project_channel_id:
            return graph.project_channel_id  # Already has one

        if not self.channel_manager:
            logger.warning("[Executor] No channel_manager, dispatching to main channel")
            return None

        # Need guild from the main channel
        main_channel = self.bot.get_channel(graph.channel_id)
        if not main_channel or not hasattr(main_channel, "guild"):
            logger.warning("[Executor] Cannot resolve guild for project channel")
            return None

        channel = await self.channel_manager.create_project_channel(
            guild=main_channel.guild,
            graph_id=graph.id,
            goal=graph.goal,
        )
        if channel:
            graph.project_channel_id = channel.id
            if channel.id not in self._project_channel_to_graphs:
                self._project_channel_to_graphs[channel.id] = []
            self._project_channel_to_graphs[channel.id].append(graph)
            logger.info(
                f"[Executor] Created meeting room {channel.id} for graph {graph.id}"
            )

            # Post an introductory embed in the meeting room
            try:
                embed = discord.Embed(
                    title=f"Meeting Room: {graph.goal[:80]}",
                    description=(
                        f"This channel is the workspace for plan `{graph.id}`.\n"
                        f"Agent dispatches and responses happen here.\n"
                        f"Final answers will be delivered to the main channel."
                    ),
                    color=COLOR_PLAN,
                )
                await channel.send(embed=embed)
            except discord.HTTPException:
                pass

            return channel.id
        return None

    def get_graph_by_project_channel(self, channel_id: int) -> Optional[JobGraph]:
        """Get the first active graph in a project channel (backward compat)."""
        for graph in self._project_channel_to_graphs.get(channel_id, []):
            if graph.status == GraphStatus.ACTIVE:
                return graph
        return None

    def get_graphs_by_project_channel(self, channel_id: int) -> list[JobGraph]:
        """Get all active graphs in a project channel."""
        return [
            g for g in self._project_channel_to_graphs.get(channel_id, [])
            if g.status == GraphStatus.ACTIVE
        ]

    def get_active_project_channel_ids(self) -> set[int]:
        """Get all channel IDs of active meeting rooms and departments."""
        active = {
            ch_id
            for ch_id, graphs in self._project_channel_to_graphs.items()
            if any(g.status == GraphStatus.ACTIVE for g in graphs)
        }
        return active | self._department_channel_ids

    # --------------------------------------------------
    # DEPARTMENT MANAGEMENT
    # --------------------------------------------------

    def assign_graph_to_department(self, graph: JobGraph, channel_id: int) -> None:
        """Assign a graph to an existing department channel."""
        graph.project_channel_id = channel_id
        if channel_id not in self._project_channel_to_graphs:
            self._project_channel_to_graphs[channel_id] = []
        self._project_channel_to_graphs[channel_id].append(graph)
        logger.info(
            f"[Executor] Assigned graph {graph.id} to department channel {channel_id}"
        )

    def register_department(self, channel_id: int, project_name: str) -> None:
        """Register a channel as a persistent department."""
        self._department_channel_ids.add(channel_id)
        self._department_info[channel_id] = project_name
        if channel_id not in self._project_channel_to_graphs:
            self._project_channel_to_graphs[channel_id] = []
        logger.info(
            f"[Executor] Registered department: channel={channel_id}, project={project_name}"
        )

    def unregister_department(self, channel_id: int) -> None:
        """Unregister a department channel."""
        self._department_channel_ids.discard(channel_id)
        self._department_info.pop(channel_id, None)
        logger.info(f"[Executor] Unregistered department: channel={channel_id}")

    def is_department_channel(self, channel_id: int) -> bool:
        """Check if a channel is a registered department."""
        return channel_id in self._department_channel_ids

    def get_department_info(self) -> dict[int, str]:
        """Get mapping of department channel IDs to project names."""
        return dict(self._department_info)

    async def _delete_project_channel(self, graph: JobGraph) -> None:
        """Delete a graph's meeting room channel."""
        if not graph.project_channel_id:
            return

        # Post a farewell message before deletion
        try:
            channel = self.bot.get_channel(graph.project_channel_id)
            if channel:
                status = "completed" if graph.status == GraphStatus.COMPLETED else "ended"
                await channel.send(f"Plan {status}. This channel will be deleted shortly.")
                await asyncio.sleep(3)
        except discord.HTTPException:
            pass

        if self.channel_manager:
            await self.channel_manager.delete_project_channel(graph.project_channel_id)

        # Clean up index - remove this graph and delete key if list is empty
        if graph.project_channel_id in self._project_channel_to_graphs:
            graphs = self._project_channel_to_graphs[graph.project_channel_id]
            self._project_channel_to_graphs[graph.project_channel_id] = [
                g for g in graphs if g.id != graph.id
            ]
            if not self._project_channel_to_graphs[graph.project_channel_id]:
                del self._project_channel_to_graphs[graph.project_channel_id]

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
                    # Clean up project channel mapping (but not department keys)
                    if graph.project_channel_id and graph.project_channel_id in self._project_channel_to_graphs:
                        pc_graphs = self._project_channel_to_graphs[graph.project_channel_id]
                        self._project_channel_to_graphs[graph.project_channel_id] = [
                            g for g in pc_graphs if g.id != graph.id
                        ]
                        # Only remove the key if empty AND not a department
                        if (not self._project_channel_to_graphs[graph.project_channel_id]
                                and graph.project_channel_id not in self._department_channel_ids):
                            del self._project_channel_to_graphs[graph.project_channel_id]
                    cleaned += 1
                else:
                    remaining.append(graph)
            self._graphs[channel_id] = remaining

        return cleaned

    @staticmethod
    def _now():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
