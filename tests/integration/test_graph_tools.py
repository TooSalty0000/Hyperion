"""Tests for Vega's graph orchestration tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.tools.interface import ToolContext
from shared.job_graph.models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus
from shared.job_graph.executor import JobGraphExecutor
from vega.tools.graph import (
    CreatePlanTool, AddNodesTool, UpdateNodeTool,
    CancelNodesTool, RespondToUserTool,
)
from tests.conftest import MockBot, MockChannel, MockMessage, MockThread


@pytest.fixture
def executor_with_channel():
    """Executor with a registered channel."""
    bot = MockBot()
    channel = MockChannel(id=100)
    bot.add_channel(channel)
    return JobGraphExecutor(bot=bot, agent_registry={"altair": 111, "polaris": 222}, max_timeout=300)


@pytest.fixture
def tool_ctx(executor_with_channel):
    """ToolContext with executor and trigger message."""
    msg = MockMessage(id=1, content="do stuff", channel=MockChannel(id=100))
    return ToolContext(
        job_executor=executor_with_channel,
        trigger_message=msg,
        current_channel_id=100,
        user_id=999,
        current_agent_id="vega",
    )


class TestCreatePlanTool:

    @pytest.mark.asyncio
    async def test_create_plan_happy_path(self, tool_ctx):
        tool = CreatePlanTool()
        result = await tool.execute(
            tool_ctx,
            goal="Deploy the app",
            nodes=[
                {"id": "n1", "type": "think", "description": "Analyze requirements"},
                {"id": "n2", "type": "dispatch", "description": "Run build", "agent": "altair", "dependencies": ["n1"]},
                {"id": "n3", "type": "respond", "description": "Tell user", "dependencies": ["n2"]},
            ]
        )
        assert "Plan created: 3 nodes" in result
        assert "Graph ID:" in result

        # Verify graph state
        graphs = tool_ctx.job_executor.get_all_active_graphs()
        assert len(graphs) == 1
        graph = graphs[0]
        assert graph.goal == "Deploy the app"
        assert len(graph.nodes) == 3

    @pytest.mark.asyncio
    async def test_create_plan_invalid_nodes_not_dicts(self, tool_ctx):
        tool = CreatePlanTool()
        result = await tool.execute(tool_ctx, goal="test", nodes=[1, 2, 3])
        assert "Error" in result
        assert "must be a list of objects" in result

        # No graph should have been created (thread not wasted)
        graphs = tool_ctx.job_executor.get_all_active_graphs()
        assert len(graphs) == 0

    @pytest.mark.asyncio
    async def test_create_plan_empty_nodes(self, tool_ctx):
        tool = CreatePlanTool()
        result = await tool.execute(tool_ctx, goal="test", nodes=[])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_create_plan_dispatch_without_agent(self, tool_ctx):
        tool = CreatePlanTool()
        result = await tool.execute(
            tool_ctx,
            goal="test",
            nodes=[{"id": "n1", "type": "dispatch", "description": "do something"}]
        )
        assert "requires an agent" in result

    @pytest.mark.asyncio
    async def test_create_plan_invalid_node_type(self, tool_ctx):
        tool = CreatePlanTool()
        result = await tool.execute(
            tool_ctx,
            goal="test",
            nodes=[{"id": "n1", "type": "invalid_type", "description": "bad"}]
        )
        assert "Node error" in result

    @pytest.mark.asyncio
    async def test_create_plan_no_executor(self):
        tool = CreatePlanTool()
        ctx = ToolContext()
        result = await tool.execute(ctx, goal="test", nodes=[{"id": "n1", "type": "think", "description": "t"}])
        assert "Error" in result
        assert "executor" in result.lower()

    @pytest.mark.asyncio
    async def test_create_plan_dispatches_ready_nodes(self, tool_ctx):
        """Nodes with no deps should be dispatched immediately."""
        # Add channel to bot so dispatch works
        tool_ctx.job_executor.bot.add_channel(MockChannel(id=100))

        tool = CreatePlanTool()
        result = await tool.execute(
            tool_ctx,
            goal="test",
            nodes=[
                {"id": "n1", "type": "think", "description": "think first"},
            ]
        )
        assert "dispatched" in result or "ready" in result


class TestAddNodesTool:

    @pytest.mark.asyncio
    async def test_add_nodes_to_existing_graph(self, tool_ctx):
        # First create a graph
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[{"id": "n1", "type": "think", "description": "first"}]
        )
        graphs = tool_ctx.job_executor.get_all_active_graphs()
        graph_id = graphs[0].id

        # Add nodes
        add = AddNodesTool()
        result = await add.execute(
            tool_ctx,
            graph_id=graph_id,
            nodes=[{"id": "n2", "type": "respond", "description": "reply", "dependencies": ["n1"]}]
        )
        assert "Added 1 nodes" in result
        assert len(graphs[0].nodes) == 2

    @pytest.mark.asyncio
    async def test_add_nodes_invalid_graph_id(self, tool_ctx):
        add = AddNodesTool()
        result = await add.execute(tool_ctx, graph_id="nonexistent", nodes=[])
        assert "Error" in result
        assert "not found" in result


class TestUpdateNodeTool:

    @pytest.mark.asyncio
    async def test_update_node_completed(self, tool_ctx):
        # Create graph with nodes
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[
                {"id": "n1", "type": "think", "description": "think"},
                {"id": "n2", "type": "respond", "description": "reply", "dependencies": ["n1"]},
            ]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        update = UpdateNodeTool()
        result = await update.execute(tool_ctx, graph_id=graph.id, node_id="n1", status="completed", result="thought about it")
        assert "marked completed" in result
        assert graph.nodes["n1"].status == NodeStatus.COMPLETED
        # After n1 completes, executor auto-dispatches n2 (RESPOND → RUNNING)
        assert graph.nodes["n2"].status == NodeStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_node_failed(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[{"id": "n1", "type": "dispatch", "description": "run", "agent": "altair"}]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        update = UpdateNodeTool()
        result = await update.execute(tool_ctx, graph_id=graph.id, node_id="n1", status="failed", result="timed out")
        assert "marked failed" in result
        assert graph.nodes["n1"].status == NodeStatus.FAILED

    @pytest.mark.asyncio
    async def test_update_node_goal_met(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[{"id": "n1", "type": "think", "description": "only task"}]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        update = UpdateNodeTool()
        result = await update.execute(tool_ctx, graph_id=graph.id, node_id="n1", status="completed", result="done")
        assert "ALL NODES COMPLETE" in result

    @pytest.mark.asyncio
    async def test_update_node_invalid_graph(self, tool_ctx):
        update = UpdateNodeTool()
        result = await update.execute(tool_ctx, graph_id="bad", node_id="n1", status="completed")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_update_node_invalid_node(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[{"id": "n1", "type": "think", "description": "t"}]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        update = UpdateNodeTool()
        result = await update.execute(tool_ctx, graph_id=graph.id, node_id="nope", status="completed")
        assert "Error" in result
        assert "not found" in result


class TestCancelNodesTool:

    @pytest.mark.asyncio
    async def test_cancel_pending_nodes(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[
                {"id": "n1", "type": "think", "description": "a"},
                {"id": "n2", "type": "dispatch", "description": "b", "agent": "altair", "dependencies": ["n1"]},
            ]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        cancel = CancelNodesTool()
        result = await cancel.execute(tool_ctx, graph_id=graph.id, node_ids=["n2"])
        assert "Cancelled 1/1" in result
        assert graph.nodes["n2"].status == NodeStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_node_fails(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="test",
            nodes=[{"id": "n1", "type": "think", "description": "a"}]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]

        # n1 was dispatched as THINK, so it's RUNNING
        assert graph.nodes["n1"].status == NodeStatus.RUNNING

        cancel = CancelNodesTool()
        result = await cancel.execute(tool_ctx, graph_id=graph.id, node_ids=["n1"])
        assert "Cancelled 0/1" in result


class TestRespondToUserTool:

    @pytest.mark.asyncio
    async def test_respond_sets_context_message(self, tool_ctx):
        tool = RespondToUserTool()
        result = await tool.execute(tool_ctx, message="Hello, user!")
        assert "queued" in result.lower()
        assert tool_ctx.response_message == "Hello, user!"

    @pytest.mark.asyncio
    async def test_respond_completes_graph(self, tool_ctx):
        create = CreatePlanTool()
        await create.execute(
            tool_ctx, goal="greet",
            nodes=[
                {"id": "n1", "type": "think", "description": "prepare"},
                {"id": "n2", "type": "respond", "description": "send", "dependencies": ["n1"]},
            ]
        )
        graph = tool_ctx.job_executor.get_all_active_graphs()[0]
        # Complete n1, making n2 ready
        graph.mark_completed("n1", "prepared")
        graph.mark_running("n2")

        respond = RespondToUserTool()
        await respond.execute(tool_ctx, message="Hi!", graph_id=graph.id)

        assert graph.status == GraphStatus.COMPLETED
        assert tool_ctx.response_message == "Hi!"
