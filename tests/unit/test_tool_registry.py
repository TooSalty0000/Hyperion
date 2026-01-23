"""Tests for ToolRegistry - register, execute, batch, and abort."""

import pytest
from unittest.mock import AsyncMock
from typing import List

from shared.tools.interface import Tool, ToolContext
from shared.tools.registry import ToolRegistry
from shared.llm.types import ToolCall, ToolResult, ToolParameter


# --------------------------------------------------
# Test Tool Implementations
# --------------------------------------------------

class EchoTool(Tool):
    """Simple tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the message back"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="message", type="string", description="Message to echo", required=True)]

    async def execute(self, context: ToolContext, **kwargs) -> str:
        return f"echo: {kwargs['message']}"


class AddTool(Tool):
    """Tool that adds two numbers."""

    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "Adds two numbers"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="a", type="integer", description="First number", required=True),
            ToolParameter(name="b", type="integer", description="Second number", required=True),
        ]

    async def execute(self, context: ToolContext, **kwargs) -> str:
        return str(kwargs["a"] + kwargs["b"])


class FailingTool(Tool):
    """Tool that always raises an exception."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext, **kwargs) -> str:
        raise RuntimeError("Intentional failure")


class TypeErrorTool(Tool):
    """Tool that raises TypeError (bad arguments)."""

    @property
    def name(self) -> str:
        return "bad_args"

    @property
    def description(self) -> str:
        return "Raises TypeError"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="x", type="string", description="unused", required=True)]

    async def execute(self, context: ToolContext, **kwargs) -> str:
        raise TypeError("missing required argument: 'y'")


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def ctx():
    return ToolContext(current_channel_id=100, user_id=999, current_agent_id="test")


# --------------------------------------------------
# Registration Tests
# --------------------------------------------------

class TestRegistration:

    def test_register_tool(self, registry):
        registry.register(EchoTool())
        assert registry.get("echo") is not None
        assert registry.get("echo").name == "echo"

    def test_register_multiple_tools(self, registry):
        registry.register(EchoTool())
        registry.register(AddTool())
        assert len(registry.list_all()) == 2

    def test_register_duplicate_raises(self, registry):
        registry.register(EchoTool())
        with pytest.raises(ValueError, match="Tool already registered"):
            registry.register(EchoTool())

    def test_unregister_existing(self, registry):
        registry.register(EchoTool())
        assert registry.unregister("echo") is True
        assert registry.get("echo") is None

    def test_unregister_nonexistent(self, registry):
        assert registry.unregister("ghost") is False

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_all_empty(self, registry):
        assert registry.list_all() == []

    def test_get_definitions(self, registry):
        registry.register(EchoTool())
        registry.register(AddTool())
        defs = registry.get_definitions()
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert "echo" in names
        assert "add" in names

    def test_get_definitions_includes_params(self, registry):
        registry.register(AddTool())
        defs = registry.get_definitions()
        add_def = defs[0]
        assert len(add_def.parameters) == 2
        assert add_def.parameters[0].name == "a"
        assert add_def.parameters[1].name == "b"


# --------------------------------------------------
# Execute Tests
# --------------------------------------------------

class TestExecute:

    @pytest.mark.asyncio
    async def test_execute_success(self, registry, ctx):
        registry.register(EchoTool())
        call = ToolCall(id="c1", name="echo", arguments={"message": "hello"})
        result = await registry.execute(call, ctx)
        assert result.call_id == "c1"
        assert result.name == "echo"
        assert result.result == "echo: hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry, ctx):
        registry.register(EchoTool())
        call = ToolCall(id="c2", name="unknown_tool", arguments={})
        result = await registry.execute(call, ctx)
        assert result.is_error is True
        assert "Unknown tool" in result.result
        assert "echo" in result.result  # Lists available tools

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, registry, ctx):
        registry.register(FailingTool())
        call = ToolCall(id="c3", name="fail", arguments={})
        result = await registry.execute(call, ctx)
        assert result.is_error is True
        assert "Intentional failure" in result.result

    @pytest.mark.asyncio
    async def test_execute_type_error(self, registry, ctx):
        registry.register(TypeErrorTool())
        call = ToolCall(id="c4", name="bad_args", arguments={"x": "test"})
        result = await registry.execute(call, ctx)
        assert result.is_error is True
        assert "Invalid arguments" in result.result

    @pytest.mark.asyncio
    async def test_execute_with_multiple_args(self, registry, ctx):
        registry.register(AddTool())
        call = ToolCall(id="c5", name="add", arguments={"a": 3, "b": 7})
        result = await registry.execute(call, ctx)
        assert result.result == "10"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_preserves_call_id(self, registry, ctx):
        registry.register(EchoTool())
        call = ToolCall(id="unique-id-123", name="echo", arguments={"message": "x"})
        result = await registry.execute(call, ctx)
        assert result.call_id == "unique-id-123"


# --------------------------------------------------
# Batch Execute Tests
# --------------------------------------------------

class TestBatchExecute:

    @pytest.mark.asyncio
    async def test_batch_all_succeed(self, registry, ctx):
        registry.register(EchoTool())
        registry.register(AddTool())
        calls = [
            ToolCall(id="b1", name="echo", arguments={"message": "first"}),
            ToolCall(id="b2", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="b3", name="echo", arguments={"message": "third"}),
        ]
        results = await registry.execute_batch(calls, ctx)
        assert len(results) == 3
        assert results[0].result == "echo: first"
        assert results[1].result == "3"
        assert results[2].result == "echo: third"

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, registry, ctx):
        registry.register(EchoTool())
        calls = [
            ToolCall(id=f"b{i}", name="echo", arguments={"message": str(i)})
            for i in range(5)
        ]
        results = await registry.execute_batch(calls, ctx)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.result == f"echo: {i}"

    @pytest.mark.asyncio
    async def test_batch_continues_after_error(self, registry, ctx):
        """A failing tool shouldn't stop the batch (just returns error result)."""
        registry.register(EchoTool())
        registry.register(FailingTool())
        calls = [
            ToolCall(id="b1", name="echo", arguments={"message": "before"}),
            ToolCall(id="b2", name="fail", arguments={}),
            ToolCall(id="b3", name="echo", arguments={"message": "after"}),
        ]
        results = await registry.execute_batch(calls, ctx)
        assert len(results) == 3
        assert results[0].is_error is False
        assert results[1].is_error is True
        assert results[2].is_error is False

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, registry, ctx):
        results = await registry.execute_batch([], ctx)
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_single_call(self, registry, ctx):
        registry.register(EchoTool())
        calls = [ToolCall(id="s1", name="echo", arguments={"message": "solo"})]
        results = await registry.execute_batch(calls, ctx)
        assert len(results) == 1
        assert results[0].result == "echo: solo"


# --------------------------------------------------
# Abort Check Tests
# --------------------------------------------------

class TestAbortCheck:

    @pytest.mark.asyncio
    async def test_abort_stops_remaining_tools(self, registry, ctx):
        """Abort after first tool should cancel the rest."""
        registry.register(EchoTool())

        call_count = 0

        async def abort_after_first():
            nonlocal call_count
            call_count += 1
            return "User cancelled"  # Always abort

        calls = [
            ToolCall(id="a1", name="echo", arguments={"message": "first"}),
            ToolCall(id="a2", name="echo", arguments={"message": "second"}),
            ToolCall(id="a3", name="echo", arguments={"message": "third"}),
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=abort_after_first)

        # First tool executes, abort fires before second, remaining are cancelled
        assert len(results) == 3
        assert results[0].result == "echo: first"
        assert "[ABORTED]" in results[1].result
        assert "[ABORTED]" in results[2].result

    @pytest.mark.asyncio
    async def test_abort_not_checked_before_first(self, registry, ctx):
        """Abort check is NOT called before the first tool."""
        registry.register(EchoTool())

        async def always_abort():
            return "abort now"

        calls = [ToolCall(id="a1", name="echo", arguments={"message": "runs"})]
        results = await registry.execute_batch(calls, ctx, abort_check=always_abort)
        # Single tool → no abort check happens
        assert len(results) == 1
        assert results[0].result == "echo: runs"

    @pytest.mark.asyncio
    async def test_abort_returns_none_continues(self, registry, ctx):
        """Abort check returning None means continue."""
        registry.register(EchoTool())

        async def never_abort():
            return None  # Don't abort

        calls = [
            ToolCall(id="a1", name="echo", arguments={"message": "one"}),
            ToolCall(id="a2", name="echo", arguments={"message": "two"}),
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=never_abort)
        assert len(results) == 2
        assert all(not r.is_error for r in results)

    @pytest.mark.asyncio
    async def test_abort_check_exception_is_swallowed(self, registry, ctx):
        """If abort_check raises, it's caught and execution continues."""
        registry.register(EchoTool())

        async def broken_abort():
            raise RuntimeError("abort check exploded")

        calls = [
            ToolCall(id="a1", name="echo", arguments={"message": "one"}),
            ToolCall(id="a2", name="echo", arguments={"message": "two"}),
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=broken_abort)
        assert len(results) == 2
        assert results[0].result == "echo: one"
        assert results[1].result == "echo: two"

    @pytest.mark.asyncio
    async def test_abort_includes_reason_in_result(self, registry, ctx):
        """The abort reason should be included in the result message."""
        registry.register(EchoTool())

        async def abort_with_reason():
            return "timeout exceeded"

        calls = [
            ToolCall(id="a1", name="echo", arguments={"message": "ok"}),
            ToolCall(id="a2", name="echo", arguments={"message": "nope"}),
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=abort_with_reason)
        assert "timeout exceeded" in results[1].result

    @pytest.mark.asyncio
    async def test_abort_conditional(self, registry, ctx):
        """Abort only after a certain number of checks."""
        registry.register(EchoTool())

        check_count = 0

        async def abort_on_third():
            nonlocal check_count
            check_count += 1
            if check_count >= 3:
                return "too many tools"
            return None

        calls = [
            ToolCall(id=f"a{i}", name="echo", arguments={"message": str(i)})
            for i in range(5)
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=abort_on_third)

        # First 4 execute (checks happen before 2nd, 3rd, 4th → abort on 4th)
        # Actually: check before index 1 (count=1, ok), before 2 (count=2, ok), before 3 (count=3, abort)
        assert len(results) == 5
        assert results[0].result == "echo: 0"
        assert results[1].result == "echo: 1"
        assert results[2].result == "echo: 2"
        assert "[ABORTED]" in results[3].result
        assert "[ABORTED]" in results[4].result

    @pytest.mark.asyncio
    async def test_no_abort_check(self, registry, ctx):
        """Without abort_check, all tools execute."""
        registry.register(EchoTool())
        calls = [
            ToolCall(id=f"n{i}", name="echo", arguments={"message": str(i)})
            for i in range(3)
        ]
        results = await registry.execute_batch(calls, ctx, abort_check=None)
        assert len(results) == 3
        assert all("echo:" in r.result for r in results)
