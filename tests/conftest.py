"""Shared test fixtures for Hyperion test suite."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Any

from shared.llm.types import LLMResponse, ToolCall, ToolResult, Message, Role
from shared.job_graph.models import JobGraph, JobNode, NodeType, NodeStatus
from shared.tools.interface import ToolContext


# --------------------------------------------------
# Mock Discord objects
# --------------------------------------------------

class MockUser:
    def __init__(self, id: int = 999, name: str = "TestUser", bot: bool = False):
        self.id = id
        self.name = name
        self.display_name = name
        self.bot = bot


class _SentinelChannel:
    """Minimal channel stub used inside MockMessage to break recursion."""
    def __init__(self, id: int = 100):
        self.id = id
        self.send = AsyncMock(return_value=None)


class MockChannel:
    def __init__(self, id: int = 100):
        self.id = id
        # Use a simple MagicMock for send return to avoid recursion
        self._send_return = MagicMock()
        self._send_return.id = id + 1000
        self._send_return.content = ""
        self._send_return.create_thread = AsyncMock(return_value=None)
        self.send = AsyncMock(return_value=self._send_return)
        self.history = self._mock_history

    def _mock_history(self, limit=10, after=None):
        return MockAsyncIterator([])


class MockThread:
    def __init__(self, id: int = 200):
        self.id = id
        self.send = AsyncMock()
        self.delete = AsyncMock()


class MockMessage:
    def __init__(self, id: int = 1, content: str = "test", author: MockUser = None, channel=None):
        self.id = id
        self.content = content
        self.author = author or MockUser()
        self.channel = channel or _SentinelChannel()
        self.embeds = []
        self.attachments = []
        self.create_thread = AsyncMock(return_value=MockThread(id=id + 5000))


class MockAsyncIterator:
    """Async iterator for mocking channel.history()."""
    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class MockBot:
    """Mock Discord bot."""
    def __init__(self):
        self.user = MockUser(id=1, name="Vega", bot=True)
        self._channels = {}

    def get_channel(self, channel_id: int):
        return self._channels.get(channel_id)

    def add_channel(self, channel: MockChannel):
        self._channels[channel.id] = channel

    def add_thread(self, thread: MockThread):
        self._channels[thread.id] = thread


# --------------------------------------------------
# Mock LLM
# --------------------------------------------------

class ScriptedLLM:
    """
    LLM that returns scripted responses in sequence.

    Usage:
        llm = ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(id="1", name="create_plan", arguments={...})]),
            LLMResponse(content="Done!"),
        ])
    """
    def __init__(self, responses: List[LLMResponse]):
        self._responses = list(responses)
        self._index = 0
        self.calls = []  # Track what was sent to the LLM

    async def complete(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        if self._index >= len(self._responses):
            return LLMResponse(content="(no more scripted responses)")
        resp = self._responses[self._index]
        self._index += 1
        return resp


# --------------------------------------------------
# Pytest fixtures
# --------------------------------------------------

@pytest.fixture
def mock_bot():
    """Mock Discord bot with a default channel."""
    bot = MockBot()
    channel = MockChannel(id=100)
    bot.add_channel(channel)
    return bot


@pytest.fixture
def mock_channel():
    return MockChannel(id=100)


@pytest.fixture
def mock_message(mock_channel):
    return MockMessage(
        id=1,
        content="Hello Vega",
        author=MockUser(id=999, name="TestUser"),
        channel=mock_channel,
    )


@pytest.fixture
def agent_registry():
    return {"altair": 111, "polaris": 222, "canopus": 333}


@pytest.fixture
def job_graph():
    """Fresh JobGraph for testing."""
    return JobGraph(goal="Test goal", trigger_message_id=1, channel_id=100, max_timeout=60)


@pytest.fixture
def tool_context():
    """Minimal ToolContext."""
    return ToolContext(
        current_channel_id=100,
        user_id=999,
        current_agent_id="vega",
    )


@pytest.fixture
def executor(mock_bot, agent_registry):
    """JobGraphExecutor with mocked bot."""
    from shared.job_graph.executor import JobGraphExecutor
    return JobGraphExecutor(bot=mock_bot, agent_registry=agent_registry, max_timeout=300)
