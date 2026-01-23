"""Integration tests for JobGraphExecutor - dispatch, timeouts, node matching, cleanup."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from shared.job_graph.executor import JobGraphExecutor
from shared.job_graph.models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus
from tests.conftest import MockBot, MockChannel, MockMessage, MockThread, MockUser


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def agent_registry():
    return {"altair": 111, "polaris": 222, "canopus": 333}


@pytest.fixture
def bot():
    bot = MockBot()
    channel = MockChannel(id=100)
    bot.add_channel(channel)
    return bot


@pytest.fixture
def executor(bot, agent_registry):
    return JobGraphExecutor(bot=bot, agent_registry=agent_registry, max_timeout=300)


@pytest.fixture
def trigger_msg(bot):
    channel = bot.get_channel(100)
    msg = MockMessage(id=500, content="Do something", author=MockUser(id=999), channel=channel)
    return msg


# --------------------------------------------------
# Graph Creation
# --------------------------------------------------

class TestCreateGraph:

    @pytest.mark.asyncio
    async def test_create_graph_basic(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test goal", trigger_message=trigger_msg)
        assert graph.goal == "test goal"
        assert graph.channel_id == 100
        assert graph.trigger_message_id == 500
        assert graph.status == GraphStatus.ACTIVE
        assert graph.max_timeout == 300

    @pytest.mark.asyncio
    async def test_create_graph_registers_in_channel(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        active = executor.get_active_graphs(100)
        assert len(active) == 1
        assert active[0].id == graph.id

    @pytest.mark.asyncio
    async def test_create_graph_creates_thread(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="my plan", trigger_message=trigger_msg)
        trigger_msg.create_thread.assert_called_once()
        call_kwargs = trigger_msg.create_thread.call_args[1]
        assert "Plan: my plan" in call_kwargs["name"]
        assert graph.thread_id is not None

    @pytest.mark.asyncio
    async def test_create_graph_registers_thread_mapping(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        found = executor.get_graph_by_thread(graph.thread_id)
        assert found is graph

    @pytest.mark.asyncio
    async def test_create_graph_truncates_long_goal(self, executor, trigger_msg):
        long_goal = "x" * 200
        await executor.create_graph(goal=long_goal, trigger_message=trigger_msg)
        call_kwargs = trigger_msg.create_thread.call_args[1]
        assert len(call_kwargs["name"]) <= 86  # "Plan: " + 80 chars

    @pytest.mark.asyncio
    async def test_multiple_graphs_same_channel(self, executor, trigger_msg):
        g1 = await executor.create_graph(goal="first", trigger_message=trigger_msg)
        g2 = await executor.create_graph(goal="second", trigger_message=trigger_msg)
        active = executor.get_active_graphs(100)
        assert len(active) == 2
        assert g1.id != g2.id


# --------------------------------------------------
# Dispatch Ready Nodes
# --------------------------------------------------

class TestDispatchReadyNodes:

    @pytest.mark.asyncio
    async def test_dispatch_think_node(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.THINK, description="reason about it")
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 1
        assert dispatched[0].id == "n1"
        assert graph.nodes["n1"].status == NodeStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dispatch_respond_node(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.RESPOND, description="answer user")
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 1
        assert graph.nodes["n1"].status == NodeStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dispatch_agent_node(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="run tests", agent="altair")
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 1
        assert graph.nodes["n1"].status == NodeStatus.RUNNING

        # Verify @mention was sent
        channel = bot.get_channel(100)
        channel.send.assert_called()
        sent_content = channel.send.call_args[0][0]
        assert "<@111>" in sent_content  # altair's ID
        assert "run tests" in sent_content

    @pytest.mark.asyncio
    async def test_dispatch_unknown_agent_fails(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="go", agent="unknown_bot")
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 0
        assert graph.nodes["n1"].status == NodeStatus.FAILED
        assert "Unknown agent" in graph.nodes["n1"].error

    @pytest.mark.asyncio
    async def test_dispatch_agent_no_channel_fails(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        # Set channel_id to one the bot doesn't have
        graph.channel_id = 99999
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="go", agent="altair")
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 0
        assert graph.nodes["n1"].status == NodeStatus.FAILED

    @pytest.mark.asyncio
    async def test_dispatch_no_agent_on_dispatch_node_fails(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="go", agent=None)
        graph.add_node(node)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 0
        assert graph.nodes["n1"].status == NodeStatus.FAILED
        assert "No agent" in graph.nodes["n1"].error

    @pytest.mark.asyncio
    async def test_dispatch_skips_pending_nodes(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        n1 = JobNode(id="n1", type=NodeType.THINK, description="first")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="second", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        dispatched = await executor.dispatch_ready_nodes(graph)
        # Only n1 is READY; n2 is PENDING
        assert len(dispatched) == 1
        assert dispatched[0].id == "n1"
        assert graph.nodes["n2"].status == NodeStatus.PENDING

    @pytest.mark.asyncio
    async def test_dispatch_multiple_ready_nodes(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="task A", agent="altair")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="task B", agent="polaris")
        graph.add_node(n1)
        graph.add_node(n2)

        dispatched = await executor.dispatch_ready_nodes(graph)
        assert len(dispatched) == 2
        assert graph.nodes["n1"].status == NodeStatus.RUNNING
        assert graph.nodes["n2"].status == NodeStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dispatch_sets_message_id(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="go", agent="altair")
        graph.add_node(node)

        await executor.dispatch_ready_nodes(graph)
        # MockChannel.send returns a mock with id=channel_id+1000
        assert graph.nodes["n1"].message_id is not None


# --------------------------------------------------
# Timeout Detection
# --------------------------------------------------

class TestTimeouts:

    @pytest.mark.asyncio
    async def test_check_timeouts_detects_expired(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="slow", agent="altair", timeout=10)
        graph.add_node(node)
        graph.mark_running("n1")

        # Simulate time passing
        graph.nodes["n1"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=20)

        timed_out = executor.check_timeouts()
        assert len(timed_out) == 1
        assert timed_out[0][1].id == "n1"
        assert graph.nodes["n1"].status == NodeStatus.FAILED
        assert "Timed out" in graph.nodes["n1"].error

    @pytest.mark.asyncio
    async def test_check_timeouts_ignores_within_limit(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="fast", agent="altair", timeout=120)
        graph.add_node(node)
        graph.mark_running("n1")
        # dispatched_at is set to now by mark_running, so it's within timeout

        timed_out = executor.check_timeouts()
        assert len(timed_out) == 0

    @pytest.mark.asyncio
    async def test_check_timeouts_ignores_completed_graphs(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        graph.status = GraphStatus.COMPLETED
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="done", agent="altair", timeout=10)
        graph.add_node(node)
        graph.mark_running("n1")
        graph.nodes["n1"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=20)

        timed_out = executor.check_timeouts()
        assert len(timed_out) == 0

    @pytest.mark.asyncio
    async def test_check_timeouts_multiple_nodes(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="slow1", agent="altair", timeout=10)
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="slow2", agent="polaris", timeout=10)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.mark_running("n1")
        graph.mark_running("n2")
        graph.nodes["n1"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=20)
        graph.nodes["n2"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=20)

        timed_out = executor.check_timeouts()
        assert len(timed_out) == 2


# --------------------------------------------------
# Find Node for Agent Response
# --------------------------------------------------

class TestFindNodeForAgentResponse:

    @pytest.mark.asyncio
    async def test_find_running_dispatch_node(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="task", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        result = executor.find_node_for_agent_response("altair", 100)
        assert result is not None
        assert result[0].id == graph.id
        assert result[1].id == "n1"

    @pytest.mark.asyncio
    async def test_find_returns_none_for_wrong_agent(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="task", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        result = executor.find_node_for_agent_response("polaris", 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_returns_none_for_wrong_channel(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="task", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        result = executor.find_node_for_agent_response("altair", 999)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_returns_oldest_when_multiple(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="first", agent="altair")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="second", agent="altair")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.mark_running("n1")
        # Make n1 dispatched earlier
        graph.nodes["n1"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        graph.mark_running("n2")

        result = executor.find_node_for_agent_response("altair", 100)
        assert result[1].id == "n1"  # Oldest first

    @pytest.mark.asyncio
    async def test_find_ignores_completed_nodes(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="task", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")

        result = executor.find_node_for_agent_response("altair", 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_ignores_think_nodes(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.THINK, description="internal")
        graph.add_node(node)
        graph.mark_running("n1")

        result = executor.find_node_for_agent_response("altair", 100)
        assert result is None


# --------------------------------------------------
# Graph Context
# --------------------------------------------------

class TestGraphContext:

    @pytest.mark.asyncio
    async def test_empty_context(self, executor):
        assert executor.get_all_graphs_context() == ""

    @pytest.mark.asyncio
    async def test_context_includes_active_graphs(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test goal", trigger_message=trigger_msg)
        node = JobNode(id="n1", type=NodeType.THINK, description="think hard")
        graph.add_node(node)

        context = executor.get_all_graphs_context()
        assert "ACTIVE JOB GRAPHS" in context
        assert "test goal" in context
        assert "think hard" in context

    @pytest.mark.asyncio
    async def test_context_excludes_completed_graphs(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="done goal", trigger_message=trigger_msg)
        graph.status = GraphStatus.COMPLETED

        context = executor.get_all_graphs_context()
        assert context == ""


# --------------------------------------------------
# Complete / Fail Graph
# --------------------------------------------------

class TestGraphLifecycle:

    @pytest.mark.asyncio
    async def test_complete_graph(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        thread = MockThread(id=graph.thread_id)
        bot.add_thread(thread)

        await executor.complete_graph(graph)
        assert graph.status == GraphStatus.COMPLETED
        # Thread should be deleted
        thread.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_fail_graph(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        thread = MockThread(id=graph.thread_id)
        bot.add_thread(thread)

        await executor.fail_graph(graph)
        assert graph.status == GraphStatus.FAILED
        thread.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_graph_cleans_thread_mapping(self, executor, trigger_msg, bot):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        thread_id = graph.thread_id
        thread = MockThread(id=thread_id)
        bot.add_thread(thread)

        await executor.complete_graph(graph)
        assert executor.get_graph_by_thread(thread_id) is None

    @pytest.mark.asyncio
    async def test_complete_graph_no_thread(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="test", trigger_message=trigger_msg)
        graph.thread_id = None  # Simulate failed thread creation

        # Should not raise
        await executor.complete_graph(graph)
        assert graph.status == GraphStatus.COMPLETED


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

class TestCleanup:

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_completed(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="old", trigger_message=trigger_msg)
        graph.status = GraphStatus.COMPLETED
        graph.created_at = datetime.now(timezone.utc) - timedelta(minutes=60)

        cleaned = executor.cleanup_completed(max_age_minutes=30)
        assert cleaned == 1
        assert executor.get_active_graphs(100) == []

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_completed(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="recent", trigger_message=trigger_msg)
        graph.status = GraphStatus.COMPLETED
        graph.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        cleaned = executor.cleanup_completed(max_age_minutes=30)
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active_graphs(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="active", trigger_message=trigger_msg)
        graph.created_at = datetime.now(timezone.utc) - timedelta(minutes=60)
        # Status stays ACTIVE

        cleaned = executor.cleanup_completed(max_age_minutes=30)
        assert cleaned == 0
        assert len(executor.get_active_graphs(100)) == 1

    @pytest.mark.asyncio
    async def test_cleanup_removes_thread_mapping(self, executor, trigger_msg):
        graph = await executor.create_graph(goal="old", trigger_message=trigger_msg)
        thread_id = graph.thread_id
        graph.status = GraphStatus.COMPLETED
        graph.created_at = datetime.now(timezone.utc) - timedelta(minutes=60)

        executor.cleanup_completed(max_age_minutes=30)
        assert executor.get_graph_by_thread(thread_id) is None
