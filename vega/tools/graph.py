"""Job Graph tools for Vega's DAG-based orchestration.

These tools allow Vega's LLM to create, modify, and manage job graphs
for orchestrating work across agents.
"""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext
from shared.job_graph.models import JobNode, JobGraph, NodeType, NodeStatus

logger = logging.getLogger(__name__)


NODE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Short unique ID like 'n1', 'n2'"},
        "type": {"type": "string", "enum": ["think", "dispatch", "respond"], "description": "Node type"},
        "description": {"type": "string", "description": "What this node does"},
        "agent": {"type": "string", "enum": ["altair", "polaris", "canopus"], "description": "Target agent (required for dispatch)"},
        "dependencies": {"type": "array", "items": {"type": "string"}, "description": "Node IDs that must complete first"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
    },
    "required": ["id", "type", "description"],
}


class CreatePlanTool(Tool):
    """
    Create a new execution plan as a DAG of job nodes.

    Call this when you need to orchestrate work across agents.
    Even simple tasks should use this - a single-node plan is valid.
    You can add more nodes later if the task becomes complex.
    """

    @property
    def name(self) -> str:
        return "create_plan"

    @property
    def description(self) -> str:
        return (
            "Create a new execution plan as a DAG (directed acyclic graph) of tasks. "
            "Each node represents a unit of work: 'think' (reason internally), "
            "'dispatch' (send task to an agent), or 'respond' (reply to user). "
            "Nodes can depend on other nodes. Available agents: altair (CLI/code), "
            "polaris (calendar), canopus (web browser). "
            "Set appropriate timeouts per node (30s for quick checks, 300s for builds)."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="goal",
                type="string",
                description="The overall goal this plan achieves",
                required=True,
            ),
            ToolParameter(
                name="nodes",
                type="array",
                description="List of job nodes for the plan",
                required=True,
                items=NODE_ITEM_SCHEMA,
            ),
        ]

    async def execute(self, context: ToolContext, goal: str, nodes: list) -> str:
        """Create a plan and register it with the executor."""
        executor = context.job_executor
        if not executor:
            return "Error: Job executor not available."

        trigger_message = context.trigger_message
        if not trigger_message:
            return "Error: No trigger message available for thread creation."

        # Validate nodes are dicts before creating the graph/thread
        if not nodes or not isinstance(nodes[0], dict):
            return (
                "Error: 'nodes' must be a list of objects, each with keys: "
                "id (string), type ('think'|'dispatch'|'respond'), description (string), "
                "agent (string, required for dispatch), dependencies (list of IDs), "
                "timeout (integer seconds). "
                "Example: [{\"id\": \"n1\", \"type\": \"think\", \"description\": \"Reason about the request\"}]"
            )

        # Create the graph (and thread)
        graph = await executor.create_graph(goal, trigger_message)

        # Add nodes
        errors = []
        for node_data in nodes:
            if not isinstance(node_data, dict):
                errors.append(f"Skipped non-object node: {node_data}")
                continue

            try:
                node_id = node_data.get("id", f"n{len(graph.nodes) + 1}")
                node_type_str = node_data.get("type", "think")
                node_type = NodeType(node_type_str)

                node = JobNode(
                    id=node_id,
                    type=node_type,
                    description=node_data.get("description", ""),
                    agent=node_data.get("agent"),
                    dependencies=node_data.get("dependencies", []),
                    timeout=node_data.get("timeout", 120),
                )

                # Validate dispatch nodes have agents
                if node_type == NodeType.DISPATCH and not node.agent:
                    errors.append(f"Node '{node_id}': dispatch type requires an agent")
                    continue

                graph.add_node(node)

            except (ValueError, KeyError) as e:
                errors.append(f"Node error: {e}")

        if errors:
            error_str = "; ".join(errors)
            return f"Plan created with {len(graph.nodes)} nodes. Errors: {error_str}"

        # Post the plan embed to the thread
        await executor._post_plan_embed(graph)

        # Dispatch any immediately ready nodes
        dispatched = await executor.dispatch_ready_nodes(graph)

        # Signal that we dispatched to external agents (Vega should yield and wait)
        has_agent_dispatch = any(n.type == NodeType.DISPATCH for n in dispatched)
        if has_agent_dispatch:
            context.dispatched_to_agents = True

        ready_count = len(graph.get_ready_nodes())
        dispatch_str = f", {len(dispatched)} dispatched" if dispatched else ""

        return (
            f"Plan created: {len(graph.nodes)} nodes{dispatch_str}, "
            f"{ready_count} ready. Graph ID: {graph.id}"
        )


class AddNodesTool(Tool):
    """
    Add new nodes to an active job graph.

    Use this when you realize more work is needed during execution.
    New nodes can depend on existing nodes (completed or pending).
    """

    @property
    def name(self) -> str:
        return "add_nodes"

    @property
    def description(self) -> str:
        return (
            "Add new nodes to an active job graph. Use this when you need to extend "
            "the plan with more work. New nodes can depend on existing nodes."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="graph_id",
                type="string",
                description="ID of the graph to add nodes to",
                required=True,
            ),
            ToolParameter(
                name="nodes",
                type="array",
                description="List of new nodes to add (same format as create_plan)",
                required=True,
                items=NODE_ITEM_SCHEMA,
            ),
        ]

    async def execute(self, context: ToolContext, graph_id: str, nodes: list) -> str:
        """Add nodes to an existing graph."""
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

        added = 0
        for node_data in nodes:
            try:
                node_id = node_data.get("id", f"n{len(graph.nodes) + 1}")
                node_type = NodeType(node_data.get("type", "think"))

                node = JobNode(
                    id=node_id,
                    type=node_type,
                    description=node_data.get("description", ""),
                    agent=node_data.get("agent"),
                    dependencies=node_data.get("dependencies", []),
                    timeout=node_data.get("timeout", 120),
                )

                graph.add_node(node)
                added += 1
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to add node: {e}")

        # Dispatch newly ready nodes
        dispatched = await executor.dispatch_ready_nodes(graph)

        # Signal that we dispatched to external agents (Vega should yield and wait)
        has_agent_dispatch = any(n.type == NodeType.DISPATCH for n in dispatched)
        if has_agent_dispatch:
            context.dispatched_to_agents = True

        return f"Added {added} nodes to graph {graph_id}. {len(dispatched)} dispatched."


class UpdateNodeTool(Tool):
    """
    Update the status of a node in a job graph.

    Use this to mark nodes as completed, failed, or cancelled.
    For think/respond nodes, Vega marks them herself after reasoning.
    """

    @property
    def name(self) -> str:
        return "update_node"

    @property
    def description(self) -> str:
        return (
            "Update the status of a node. Use 'completed' when work is done, "
            "'failed' if something went wrong, 'cancelled' to skip it. "
            "Include a result/error message for context."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="graph_id",
                type="string",
                description="ID of the graph",
                required=True,
            ),
            ToolParameter(
                name="node_id",
                type="string",
                description="ID of the node to update",
                required=True,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="New status: 'completed', 'failed', or 'cancelled'",
                required=True,
                enum=["completed", "failed", "cancelled"],
            ),
            ToolParameter(
                name="result",
                type="string",
                description="Result text (for completed) or error message (for failed)",
                required=False,
            ),
        ]

    async def execute(
        self, context: ToolContext, graph_id: str, node_id: str,
        status: str, result: str = ""
    ) -> str:
        """Update a node's status."""
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
            return f"Error: Graph '{graph_id}' not found."

        node = graph.nodes.get(node_id)
        if not node:
            return f"Error: Node '{node_id}' not found in graph '{graph_id}'."

        if status == "completed":
            graph.mark_completed(node_id, result)
        elif status == "failed":
            graph.mark_failed(node_id, result)
        elif status == "cancelled":
            graph.cancel_node(node_id)
        else:
            return f"Error: Invalid status '{status}'."

        # Dispatch any newly ready nodes
        dispatched = await executor.dispatch_ready_nodes(graph)

        # Signal that we dispatched to external agents (Vega should yield and wait)
        has_agent_dispatch = any(n.type == NodeType.DISPATCH for n in dispatched)
        if has_agent_dispatch:
            context.dispatched_to_agents = True

        # Post node update embed to thread
        await executor._post_node_update_embed(graph, node, status, result)

        # Check if graph is done
        if graph.is_goal_met():
            await executor.complete_graph(graph)
            return f"Node '{node_id}' marked {status}. ALL NODES COMPLETE - goal achieved!"

        dispatch_info = f" {len(dispatched)} new nodes dispatched." if dispatched else ""
        return f"Node '{node_id}' marked {status}.{dispatch_info}"


class CancelNodesTool(Tool):
    """Cancel one or more pending/ready nodes in a graph."""

    @property
    def name(self) -> str:
        return "cancel_nodes"

    @property
    def description(self) -> str:
        return "Cancel one or more pending nodes in a graph. Only works on nodes not yet running."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="graph_id",
                type="string",
                description="ID of the graph",
                required=True,
            ),
            ToolParameter(
                name="node_ids",
                type="array",
                description="List of node IDs to cancel",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, graph_id: str, node_ids: list) -> str:
        """Cancel nodes."""
        executor = context.job_executor
        if not executor:
            return "Error: Job executor not available."

        graph = None
        for g in executor.get_all_active_graphs():
            if g.id == graph_id:
                graph = g
                break

        if not graph:
            return f"Error: Graph '{graph_id}' not found."

        cancelled = 0
        for nid in node_ids:
            if graph.cancel_node(nid):
                cancelled += 1

        return f"Cancelled {cancelled}/{len(node_ids)} nodes."


class RespondToUserTool(Tool):
    """
    Send a response to the user in the main channel.

    Use this when the goal is achieved and you want to deliver the final answer.
    This marks the associated respond-node as completed if one exists.
    """

    @property
    def name(self) -> str:
        return "respond_to_user"

    @property
    def description(self) -> str:
        return (
            "Send a final response to the user in the main channel (not the thread). "
            "Use this when you've gathered enough information to answer the user's request."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type="string",
                description="The response message to send to the user",
                required=True,
            ),
            ToolParameter(
                name="graph_id",
                type="string",
                description="ID of the graph this response is for (to mark complete)",
                required=False,
            ),
        ]

    async def execute(self, context: ToolContext, message: str, graph_id: str = None) -> str:
        """Send response to user."""
        executor = context.job_executor
        if not executor:
            return "Error: Job executor not available."

        # Find the graph and complete it
        if graph_id:
            for g in executor.get_all_active_graphs():
                if g.id == graph_id:
                    # Mark any respond nodes as completed
                    for node in g.nodes.values():
                        if node.type == NodeType.RESPOND and node.status == NodeStatus.RUNNING:
                            g.mark_completed(node.id, message[:200])
                    # Complete the graph (posts embed + deletes thread)
                    await executor.complete_graph(g)
                    break

        # The actual sending is handled by the cog after process() returns
        context.response_message = message
        return f"Response queued for delivery to user ({len(message)} chars)."


def get_graph_tools() -> list[Tool]:
    """Get all graph management tools."""
    return [
        CreatePlanTool(),
        AddNodesTool(),
        UpdateNodeTool(),
        CancelNodesTool(),
        RespondToUserTool(),
    ]
