"""Integration tests for VegaAgent.process() - full LLM loop with ScriptedLLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.base_agent import AgentContext, AgentResponse
from shared.llm.types import LLMResponse, ToolCall, ToolResult, Message, Role
from shared.job_graph.executor import JobGraphExecutor
from shared.job_graph.models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus
from tests.conftest import MockBot, MockChannel, MockMessage, MockThread, MockUser, ScriptedLLM


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
    return MockMessage(id=500, content="user request", author=MockUser(id=999), channel=channel)


@pytest.fixture
def context():
    return AgentContext(
        channel_id=100,
        user_id=999,
        message_content="Hello Vega",
        conversation_id="conv-1",
    )


def make_vega(llm):
    """Create a VegaAgent with a given LLM and no optional dependencies."""
    from vega.agents.vega import VegaAgent
    return VegaAgent(llm=llm)


# --------------------------------------------------
# Simple Responses (No Tool Calls)
# --------------------------------------------------

class TestSimpleResponse:

    @pytest.mark.asyncio
    async def test_direct_text_response(self, context):
        """LLM returns text with no tool calls → immediate response."""
        llm = ScriptedLLM([
            LLMResponse(content="Hello! How can I help?"),
        ])
        agent = make_vega(llm)
        response = await agent.process(context)

        assert response.content == "Hello! How can I help?"
        assert response.agent_name == "Vega"
        assert response.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_empty_content_response(self, context):
        """LLM returns empty content with no tool calls."""
        llm = ScriptedLLM([
            LLMResponse(content=""),
        ])
        agent = make_vega(llm)
        response = await agent.process(context)

        assert response.content == ""
        assert response.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_none_content_response(self, context):
        """LLM returns None content (happens when all work is done via tools)."""
        llm = ScriptedLLM([
            LLMResponse(content=None),
        ])
        agent = make_vega(llm)
        response = await agent.process(context)

        assert response.content is None
        assert response.tool_calls_made == 0


# --------------------------------------------------
# Plan Creation Flow
# --------------------------------------------------

class TestPlanCreation:

    @pytest.mark.asyncio
    async def test_create_plan_then_respond(self, context, executor, trigger_msg):
        """LLM creates a plan, then immediately responds to user."""
        llm = ScriptedLLM([
            # First: create a simple think→respond plan
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "greet user",
                    "nodes": [
                        {"id": "t1", "type": "think", "description": "think about greeting"},
                        {"id": "r1", "type": "respond", "description": "say hi", "dependencies": ["t1"]},
                    ]
                })
            ]),
            # Second: update think node and respond
            LLMResponse(tool_calls=[
                ToolCall(id="tc2", name="update_node", arguments={
                    "graph_id": "__GRAPH_ID__",  # Will be filled dynamically
                    "node_id": "t1",
                    "status": "completed",
                    "result": "thought about it",
                }),
                ToolCall(id="tc3", name="respond_to_user", arguments={
                    "message": "Hey there! What's up?",
                    "graph_id": "__GRAPH_ID__",
                }),
            ]),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        # The response should be from respond_to_user
        assert response.content == "Hey there! What's up?"
        assert response.tool_calls_made == 3  # create_plan + update_node + respond_to_user

    @pytest.mark.asyncio
    async def test_create_plan_dispatches_to_agent(self, context, executor, trigger_msg, bot):
        """LLM creates a plan with a dispatch node → agent gets @mentioned."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "check git status",
                    "nodes": [
                        {"id": "d1", "type": "dispatch", "description": "run git status", "agent": "altair", "timeout": 60},
                    ]
                })
            ]),
            # After plan is created, LLM gets no more calls (dispatched and waiting)
            LLMResponse(content=None),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        # Altair should have been @mentioned
        channel = bot.get_channel(100)
        channel.send.assert_called()
        sent_content = channel.send.call_args[0][0]
        assert "<@111>" in sent_content  # altair
        assert "git status" in sent_content

        # Vega should yield (None content) and NOT invoke LLM a second time
        assert response.content is None
        assert len(llm.calls) == 1  # Only 1 LLM call, then yield

    @pytest.mark.asyncio
    async def test_think_only_plan_does_not_yield(self, context, executor, trigger_msg):
        """Plans with only THINK nodes should NOT yield — Vega handles them internally."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "just think",
                    "nodes": [
                        {"id": "t1", "type": "think", "description": "ponder"},
                    ]
                })
            ]),
            # Vega should continue to iteration 2 (think nodes are internal)
            LLMResponse(tool_calls=[
                ToolCall(id="tc2", name="respond_to_user", arguments={
                    "message": "I pondered and here's the answer.",
                    "graph_id": "__FILL__",
                }),
            ]),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        # Should NOT yield — should continue and respond
        assert response.content == "I pondered and here's the answer."
        assert len(llm.calls) == 2  # Both iterations ran


# --------------------------------------------------
# respond_to_user Short-Circuit
# --------------------------------------------------

class TestRespondToUserShortCircuit:

    @pytest.mark.asyncio
    async def test_respond_to_user_stops_loop(self, context, executor, trigger_msg):
        """When respond_to_user is called, the loop stops immediately."""
        llm = ScriptedLLM([
            # Create plan and respond in same batch
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "quick answer",
                    "nodes": [
                        {"id": "t1", "type": "think", "description": "easy"},
                    ]
                }),
                ToolCall(id="tc2", name="respond_to_user", arguments={
                    "message": "Here's your answer!",
                    "graph_id": "__WILL_USE_FIRST__",
                }),
            ]),
            # This should NEVER be reached
            LLMResponse(content="THIS SHOULD NOT APPEAR"),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        assert response.content == "Here's your answer!"
        # Only 1 LLM call was made (not 2)
        assert len(llm.calls) == 1


# --------------------------------------------------
# Multi-Iteration Loop
# --------------------------------------------------

class TestMultiIterationLoop:

    @pytest.mark.asyncio
    async def test_two_iteration_tool_use(self, context):
        """LLM uses tools across multiple iterations before final text."""
        llm = ScriptedLLM([
            # Iteration 1: create plan
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "two step",
                    "nodes": [
                        {"id": "t1", "type": "think", "description": "step 1"},
                        {"id": "t2", "type": "think", "description": "step 2", "dependencies": ["t1"]},
                    ]
                })
            ]),
            # Iteration 2: complete t1
            LLMResponse(tool_calls=[
                ToolCall(id="tc2", name="update_node", arguments={
                    "graph_id": "__FILL__",
                    "node_id": "t1",
                    "status": "completed",
                    "result": "step 1 done",
                })
            ]),
            # Iteration 3: complete t2 and respond
            LLMResponse(tool_calls=[
                ToolCall(id="tc3", name="update_node", arguments={
                    "graph_id": "__FILL__",
                    "node_id": "t2",
                    "status": "completed",
                    "result": "step 2 done",
                }),
                ToolCall(id="tc4", name="respond_to_user", arguments={
                    "message": "All done!",
                    "graph_id": "__FILL__",
                }),
            ]),
        ])
        agent = make_vega(llm)
        executor_local = JobGraphExecutor(
            bot=MockBot(), agent_registry={}, max_timeout=300
        )
        trigger = MockMessage(id=1, content="test")
        response = await agent.process(context, job_executor=executor_local, trigger_message=trigger)

        assert response.content == "All done!"
        assert response.tool_calls_made == 4
        assert len(llm.calls) == 3  # 3 LLM iterations


# --------------------------------------------------
# Max Iterations Safety
# --------------------------------------------------

class TestMaxIterations:

    @pytest.mark.asyncio
    async def test_max_iterations_returns_error(self, context):
        """If LLM keeps calling tools forever, we stop after 10 iterations."""
        # Create an LLM that always returns tool calls
        infinite_responses = [
            LLMResponse(tool_calls=[
                ToolCall(id=f"tc{i}", name="create_plan", arguments={
                    "goal": f"plan {i}",
                    "nodes": [{"id": f"n{i}", "type": "think", "description": f"think {i}"}]
                })
            ])
            for i in range(15)  # More than max_iterations (10)
        ]
        llm = ScriptedLLM(infinite_responses)
        agent = make_vega(llm)
        executor_local = JobGraphExecutor(
            bot=MockBot(), agent_registry={}, max_timeout=300
        )
        trigger = MockMessage(id=1, content="test")
        response = await agent.process(context, job_executor=executor_local, trigger_message=trigger)

        # Should stop after 10 iterations
        assert len(llm.calls) == 10
        assert response.tool_calls_made == 10
        # Status should indicate error (since no respond_to_user was called)
        assert response.status == "error"
        assert "Max iterations" in response.error


# --------------------------------------------------
# Error Handling
# --------------------------------------------------

class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_llm_exception_returns_error_response(self, context):
        """If LLM.complete() raises, we get an error response."""
        class ExplodingLLM:
            async def complete(self, messages, tools=None, **kwargs):
                raise RuntimeError("LLM is down")

        agent = make_vega(ExplodingLLM())
        response = await agent.process(context)

        assert "error" in response.content.lower() or "error" in response.status
        assert response.status == "error"
        assert "LLM is down" in response.error

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_in_result(self, context, executor, trigger_msg):
        """If LLM calls a non-existent tool, the registry returns an error result and loop continues."""
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="nonexistent_tool", arguments={"x": 1})
            ]),
            # After error, LLM just responds
            LLMResponse(content="Oops, let me try differently."),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        # The agent should recover and return the second response
        assert response.content == "Oops, let me try differently."
        assert response.tool_calls_made == 1  # The failed call still counts


# --------------------------------------------------
# Graph Context in System Prompt
# --------------------------------------------------

class TestGraphContext:

    @pytest.mark.asyncio
    async def test_graph_context_included_in_prompt(self, context, executor, trigger_msg):
        """When executor has active graphs, context is included in system prompt."""
        # Pre-create a graph
        graph = await executor.create_graph(goal="existing plan", trigger_message=trigger_msg)
        node = JobNode(id="x1", type=NodeType.DISPATCH, description="running task", agent="altair")
        graph.add_node(node)
        graph.mark_running("x1")

        llm = ScriptedLLM([
            LLMResponse(content="I see the active plan."),
        ])
        agent = make_vega(llm)
        response = await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        # Check that the LLM received graph context in system prompt
        system_msg = llm.calls[0]["messages"][0]
        assert "ACTIVE JOB GRAPH" in system_msg.content
        assert "existing plan" in system_msg.content
        assert "running task" in system_msg.content

    @pytest.mark.asyncio
    async def test_no_graph_context_when_empty(self, context, executor, trigger_msg):
        """When no active graphs, context is not cluttered."""
        llm = ScriptedLLM([
            LLMResponse(content="Fresh start."),
        ])
        agent = make_vega(llm)
        await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        system_msg = llm.calls[0]["messages"][0]
        assert "ACTIVE JOB GRAPH" not in system_msg.content


# --------------------------------------------------
# Tool Context Wiring
# --------------------------------------------------

class TestToolContextWiring:

    @pytest.mark.asyncio
    async def test_tool_context_has_executor(self, context, executor, trigger_msg):
        """Tools receive the job_executor through context."""
        tool_contexts_seen = []

        # Patch the tool registry to capture context
        from shared.tools.registry import ToolRegistry
        original_execute = ToolRegistry.execute

        async def capture_execute(self, call, ctx):
            tool_contexts_seen.append(ctx)
            return await original_execute(self, call, ctx)

        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "test",
                    "nodes": [{"id": "n1", "type": "think", "description": "t"}]
                })
            ]),
            LLMResponse(content="done"),
        ])
        agent = make_vega(llm)

        with patch.object(ToolRegistry, 'execute', capture_execute):
            await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        assert len(tool_contexts_seen) > 0
        assert tool_contexts_seen[0].job_executor is executor
        assert tool_contexts_seen[0].trigger_message is trigger_msg

    @pytest.mark.asyncio
    async def test_tool_context_has_user_info(self, context, executor, trigger_msg):
        """Tools receive user_id and channel_id."""
        tool_contexts_seen = []

        from shared.tools.registry import ToolRegistry
        original_execute = ToolRegistry.execute

        async def capture_execute(self, call, ctx):
            tool_contexts_seen.append(ctx)
            return await original_execute(self, call, ctx)

        llm = ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="tc1", name="create_plan", arguments={
                    "goal": "test",
                    "nodes": [{"id": "n1", "type": "think", "description": "t"}]
                })
            ]),
            LLMResponse(content="done"),
        ])
        agent = make_vega(llm)

        with patch.object(ToolRegistry, 'execute', capture_execute):
            await agent.process(context, job_executor=executor, trigger_message=trigger_msg)

        assert tool_contexts_seen[0].current_channel_id == 100
        assert tool_contexts_seen[0].user_id == 999
        assert tool_contexts_seen[0].current_agent_id == "vega"
