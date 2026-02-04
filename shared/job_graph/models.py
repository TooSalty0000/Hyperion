"""Job Graph models - DAG-based task orchestration."""

import uuid
import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of job nodes in the DAG."""
    THINK = "think"          # Vega reasons internally (no external dispatch)
    DISPATCH = "dispatch"    # Send @mention to an agent
    RESPOND = "respond"      # Respond to user in main channel


class NodeStatus(Enum):
    """Status of a job node."""
    PENDING = "pending"      # Waiting for dependencies
    READY = "ready"          # Dependencies met, can execute
    RUNNING = "running"      # Dispatched or thinking
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GraphStatus(Enum):
    """Status of the overall job graph."""
    ACTIVE = "active"        # In progress
    COMPLETED = "completed"  # Goal achieved
    FAILED = "failed"        # Unrecoverable failure


@dataclass
class JobNode:
    """A single node in the job DAG."""
    id: str
    type: NodeType
    description: str
    agent: Optional[str] = None          # For DISPATCH nodes: "altair", "polaris", "canopus"
    dependencies: list[str] = field(default_factory=list)  # Node IDs that must complete first
    status: NodeStatus = NodeStatus.PENDING
    timeout: int = 300                   # Seconds, set by Vega's LLM per node
    result: Optional[str] = None         # Result/output after completion
    error: Optional[str] = None          # Error message if failed
    message_id: Optional[int] = None     # Discord message ID of dispatched mention
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dispatched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        """Whether this node is in a terminal state."""
        return self.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.CANCELLED)

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Seconds since dispatch, or None if not dispatched."""
        if not self.dispatched_at:
            return None
        now = datetime.now(timezone.utc)
        return (now - self.dispatched_at).total_seconds()

    @property
    def is_timed_out(self) -> bool:
        """Whether this node has exceeded its timeout."""
        elapsed = self.elapsed_seconds
        if elapsed is None:
            return False
        return elapsed > self.timeout and self.status == NodeStatus.RUNNING

    def to_summary(self) -> str:
        """Human-readable summary for thread display."""
        status_icons = {
            NodeStatus.PENDING: "⏳",
            NodeStatus.READY: "🔜",
            NodeStatus.RUNNING: "🔄",
            NodeStatus.COMPLETED: "✅",
            NodeStatus.FAILED: "❌",
            NodeStatus.CANCELLED: "⛔",
        }
        icon = status_icons.get(self.status, "❓")
        agent_str = f" → {self.agent}" if self.agent else ""
        return f"{icon} `{self.id}`: {self.description}{agent_str}"


@dataclass
class JobGraph:
    """
    A directed acyclic graph of tasks.

    Represents Vega's plan for accomplishing a user's goal.
    Nodes can be added, cancelled, or re-assigned during execution.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    nodes: dict[str, JobNode] = field(default_factory=dict)
    status: GraphStatus = GraphStatus.ACTIVE
    thread_id: Optional[int] = None       # Discord thread ID for planning visibility
    trigger_message_id: Optional[int] = None  # User's original message
    channel_id: Optional[int] = None      # Main channel where user sent the message
    project_channel_id: Optional[int] = None  # Meeting room / department channel for agent dispatches
    project_name: Optional[str] = None    # Associated project name (for departments)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_timeout: int = 600                # Overall max timeout per node (env: JOB_MAX_TIMEOUT)

    def add_node(self, node: JobNode) -> None:
        """Add a node to the graph."""
        # Cap timeout at max
        node.timeout = min(node.timeout, self.max_timeout)
        self.nodes[node.id] = node
        # Update ready status
        self._update_ready_nodes()

    def remove_node(self, node_id: str) -> None:
        """Remove a node (only if PENDING)."""
        node = self.nodes.get(node_id)
        if node and node.status == NodeStatus.PENDING:
            del self.nodes[node_id]
            # Remove from other nodes' dependencies
            for other in self.nodes.values():
                if node_id in other.dependencies:
                    other.dependencies.remove(node_id)
            self._update_ready_nodes()

    def cancel_node(self, node_id: str) -> bool:
        """Cancel a node. Only works on PENDING or READY nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        if node.status in (NodeStatus.PENDING, NodeStatus.READY):
            node.status = NodeStatus.CANCELLED
            self._update_ready_nodes()
            return True
        return False

    def mark_running(self, node_id: str, message_id: Optional[int] = None) -> None:
        """Mark a node as running (dispatched)."""
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.RUNNING
            node.dispatched_at = datetime.now(timezone.utc)
            if message_id:
                node.message_id = message_id

    def mark_completed(self, node_id: str, result: str = "") -> None:
        """Mark a node as completed with optional result."""
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.COMPLETED
            node.result = result
            node.completed_at = datetime.now(timezone.utc)
            self._update_ready_nodes()
            # Check if graph is complete
            if self.is_goal_met():
                self.status = GraphStatus.COMPLETED

    def mark_failed(self, node_id: str, error: str = "") -> None:
        """Mark a node as failed."""
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.FAILED
            node.error = error
            node.completed_at = datetime.now(timezone.utc)
            self._update_ready_nodes()

    def get_ready_nodes(self) -> list[JobNode]:
        """Get all nodes that are ready to execute (dependencies met), oldest first."""
        ready = [n for n in self.nodes.values() if n.status == NodeStatus.READY]
        ready.sort(key=lambda n: n.created_at)
        return ready

    def get_running_nodes(self) -> list[JobNode]:
        """Get all currently running nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.RUNNING]

    def get_timed_out_nodes(self) -> list[JobNode]:
        """Get running nodes that have exceeded their timeout."""
        return [n for n in self.nodes.values() if n.is_timed_out]

    def is_goal_met(self) -> bool:
        """
        Check if the goal is met.

        The goal is met when all non-cancelled nodes are completed.
        """
        active_nodes = [
            n for n in self.nodes.values()
            if n.status != NodeStatus.CANCELLED
        ]
        if not active_nodes:
            return False
        return all(n.status == NodeStatus.COMPLETED for n in active_nodes)

    def has_failures(self) -> bool:
        """Check if any nodes have failed."""
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def is_stuck(self) -> bool:
        """
        Check if the graph is stuck (no ready nodes and not complete).

        This can happen if there are circular dependencies (shouldn't exist in a DAG)
        or if all remaining nodes depend on failed nodes.
        """
        if self.is_goal_met():
            return False
        if self.get_ready_nodes():
            return False
        if self.get_running_nodes():
            return False  # Still waiting
        # No ready, no running, not complete → stuck
        return True

    def get_state_summary(self) -> str:
        """Get a human-readable summary of the graph state for the planning thread."""
        lines = [f"**Goal:** {self.goal}"]

        if not self.nodes:
            lines.append("_No tasks planned yet._")
            return "\n".join(lines)

        for node in self.nodes.values():
            lines.append(node.to_summary())

        # Status counts
        counts = {}
        for node in self.nodes.values():
            counts[node.status.value] = counts.get(node.status.value, 0) + 1
        status_str = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines.append(f"\n_{status_str}_")

        return "\n".join(lines)

    def get_context_for_llm(self) -> str:
        """
        Get the graph state formatted for LLM context.

        This is included in Vega's system prompt so she knows what's happening.
        """
        lines = [f"ACTIVE JOB GRAPH (id={self.id}, status={self.status.value}):"]
        lines.append(f"Goal: {self.goal}")
        if self.project_channel_id:
            label = "Department" if self.project_name else "Meeting Room"
            channel_ref = f"<#{self.project_channel_id}>"
            if self.project_name:
                lines.append(f"{label}: {channel_ref} (project: {self.project_name})")
            else:
                lines.append(f"{label}: {channel_ref}")
        lines.append(f"Nodes:")

        for node in self.nodes.values():
            dep_str = f" (depends on: {', '.join(node.dependencies)})" if node.dependencies else ""
            result_str = f" → result: {node.result[:100]}" if node.result else ""
            error_str = f" → error: {node.error[:100]}" if node.error else ""
            timeout_str = ""
            if node.status == NodeStatus.RUNNING and node.elapsed_seconds:
                timeout_str = f" [{node.elapsed_seconds:.0f}s/{node.timeout}s]"

            lines.append(
                f"  - {node.id} [{node.type.value}] ({node.status.value}): "
                f"{node.description}{dep_str}{timeout_str}{result_str}{error_str}"
            )

        return "\n".join(lines)

    def _update_ready_nodes(self) -> None:
        """Update PENDING nodes to READY if all their dependencies are met."""
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue

            # Check if all dependencies are completed (or cancelled)
            deps_met = all(
                self.nodes[dep_id].status in (NodeStatus.COMPLETED, NodeStatus.CANCELLED)
                for dep_id in node.dependencies
                if dep_id in self.nodes
            )

            if deps_met:
                node.status = NodeStatus.READY
