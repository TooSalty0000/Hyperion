"""Tests for the Ollama LLM provider.

Covers the prompt-based tool calling fallback pipeline:
  _build_tool_prompt → _build_response_schema → _parse_prompt_tool_response

And the provider's format/parse methods:
  format_messages, format_tools, parse_response
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from shared.llm.ollama import (
    OllamaProvider,
    _build_tool_prompt,
    _build_response_schema,
    _parse_prompt_tool_response,
    _is_retryable_error,
    _is_tool_unsupported_error,
)
from shared.llm.types import (
    Message, Role, ToolDefinition, ToolParameter,
    ToolCall, ToolResult, LLMResponse,
)


# ===========================================================================
# Helper fixtures
# ===========================================================================


def _make_tool(name="get_weather", description="Get weather", params=None):
    """Create a simple ToolDefinition for testing."""
    if params is None:
        params = [
            ToolParameter(name="city", type="string", description="City name", required=True),
        ]
    return ToolDefinition(name=name, description=description, parameters=params)


def _make_provider(**kwargs):
    """Create an OllamaProvider without connecting to a server."""
    return OllamaProvider(api_key="", model="gemma3:4b", **kwargs)


# ===========================================================================
# _build_tool_prompt
# ===========================================================================


class TestBuildToolPrompt:
    """Tests for _build_tool_prompt(tools)."""

    def test_single_tool(self):
        tools = [_make_tool()]
        prompt = _build_tool_prompt(tools)
        assert "get_weather" in prompt
        assert "Get weather" in prompt
        assert '"city"' in prompt

    def test_multiple_tools(self):
        tools = [
            _make_tool("tool_a", "Does A"),
            _make_tool("tool_b", "Does B"),
        ]
        prompt = _build_tool_prompt(tools)
        assert "tool_a" in prompt
        assert "tool_b" in prompt
        assert "Does A" in prompt
        assert "Does B" in prompt

    def test_contains_json_schema(self):
        tools = [_make_tool()]
        prompt = _build_tool_prompt(tools)
        # Should contain the full JSON schema, not just param names
        assert '"type": "object"' in prompt
        assert '"properties"' in prompt
        assert '"required"' in prompt

    def test_includes_instruction_rules(self):
        tools = [_make_tool()]
        prompt = _build_tool_prompt(tools)
        assert "RULES" in prompt
        assert "ONE tool" in prompt

    def test_complex_nested_tool(self):
        """Tools with nested array/object schemas should be fully serialized."""
        node_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["think", "dispatch"]},
            },
            "required": ["id", "type"],
        }
        tools = [ToolDefinition(
            name="create_plan",
            description="Create a plan",
            parameters=[
                ToolParameter(name="goal", type="string", description="Goal", required=True),
                ToolParameter(name="nodes", type="array", description="Nodes", required=True, items=node_schema),
            ]
        )]
        prompt = _build_tool_prompt(tools)
        assert '"enum"' in prompt
        assert '"think"' in prompt
        assert '"dispatch"' in prompt


# ===========================================================================
# _build_response_schema
# ===========================================================================


class TestBuildResponseSchema:
    """Tests for _build_response_schema(tools)."""

    def test_structure(self):
        tools = [_make_tool("foo"), _make_tool("bar")]
        schema = _build_response_schema(tools)
        assert schema["type"] == "object"
        assert "tool_calls" in schema["properties"]
        assert schema["required"] == ["tool_calls"]

    def test_tool_names_in_enum(self):
        tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
        schema = _build_response_schema(tools)
        name_enum = schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]
        assert name_enum == ["alpha", "beta", "gamma"]

    def test_required_fields_on_items(self):
        tools = [_make_tool()]
        schema = _build_response_schema(tools)
        item_schema = schema["properties"]["tool_calls"]["items"]
        assert "name" in item_schema["required"]
        assert "arguments" in item_schema["required"]

    def test_arguments_is_unconstrained_object(self):
        tools = [_make_tool()]
        schema = _build_response_schema(tools)
        args_schema = schema["properties"]["tool_calls"]["items"]["properties"]["arguments"]
        assert args_schema == {"type": "object"}

    def test_single_tool(self):
        tools = [_make_tool("only_tool")]
        schema = _build_response_schema(tools)
        enum = schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]
        assert enum == ["only_tool"]


# ===========================================================================
# _parse_prompt_tool_response
# ===========================================================================


class TestParsePromptToolResponse:
    """Tests for _parse_prompt_tool_response(text)."""

    def test_valid_tool_call(self):
        text = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}]}'
        calls, msg = _parse_prompt_tool_response(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "London"}
        assert calls[0].id == "call_0"
        assert msg is None

    def test_multiple_tool_calls(self):
        text = json.dumps({"tool_calls": [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {"y": 2}},
        ]})
        calls, msg = _parse_prompt_tool_response(text)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"
        assert calls[1].id == "call_1"

    def test_message_response(self):
        text = '{"message": "Hello, how can I help?"}'
        calls, msg = _parse_prompt_tool_response(text)
        assert calls is None
        assert msg == "Hello, how can I help?"

    def test_tool_calls_takes_priority_over_message(self):
        """If both present, tool_calls wins."""
        text = json.dumps({
            "tool_calls": [{"name": "foo", "arguments": {}}],
            "message": "ignored"
        })
        calls, msg = _parse_prompt_tool_response(text)
        assert calls is not None
        assert msg is None

    def test_markdown_fenced_json(self):
        text = '```json\n{"tool_calls": [{"name": "test", "arguments": {}}]}\n```'
        calls, msg = _parse_prompt_tool_response(text)
        assert calls is not None
        assert calls[0].name == "test"

    def test_markdown_fence_no_language(self):
        text = '```\n{"tool_calls": [{"name": "test", "arguments": {}}]}\n```'
        calls, msg = _parse_prompt_tool_response(text)
        assert calls is not None

    def test_empty_input(self):
        calls, msg = _parse_prompt_tool_response("")
        assert calls is None
        assert msg is None

    def test_none_equivalent(self):
        calls, msg = _parse_prompt_tool_response("")
        assert calls is None
        assert msg is None

    def test_invalid_json(self):
        calls, msg = _parse_prompt_tool_response("This is just text, not JSON")
        assert calls is None
        assert msg is None

    def test_empty_tool_calls_array(self):
        text = '{"tool_calls": []}'
        calls, msg = _parse_prompt_tool_response(text)
        assert calls is None  # Empty list → None

    def test_missing_name_skipped(self):
        text = json.dumps({"tool_calls": [
            {"arguments": {"x": 1}},  # No name
            {"name": "valid", "arguments": {}},
        ]})
        calls, msg = _parse_prompt_tool_response(text)
        assert len(calls) == 1
        assert calls[0].name == "valid"

    def test_missing_arguments_defaults_empty(self):
        text = '{"tool_calls": [{"name": "foo"}]}'
        calls, msg = _parse_prompt_tool_response(text)
        assert calls[0].arguments == {}

    def test_non_dict_items_skipped(self):
        text = json.dumps({"tool_calls": ["not_a_dict", {"name": "ok", "arguments": {}}]})
        calls, msg = _parse_prompt_tool_response(text)
        assert len(calls) == 1
        assert calls[0].name == "ok"


# ===========================================================================
# OllamaProvider.format_messages
# ===========================================================================


class TestFormatMessages:
    """Tests for OllamaProvider.format_messages()."""

    def test_system_message(self):
        provider = _make_provider()
        msgs = [Message(role=Role.SYSTEM, content="You are helpful")]
        result = provider.format_messages(msgs)
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_user_message(self):
        provider = _make_provider()
        msgs = [Message(role=Role.USER, content="Hello")]
        result = provider.format_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        provider = _make_provider()
        msgs = [Message(role=Role.ASSISTANT, content="Hi there")]
        result = provider.format_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_with_tool_calls(self):
        provider = _make_provider()
        msgs = [Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="c1", name="foo", arguments={"x": 1})]
        )]
        result = provider.format_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"] == [{"function": {"name": "foo", "arguments": {"x": 1}}}]

    def test_tool_results_become_separate_messages(self):
        provider = _make_provider()
        msgs = [Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(call_id="c1", name="foo", result="result1"),
                ToolResult(call_id="c2", name="bar", result="result2"),
            ]
        )]
        result = provider.format_messages(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "tool", "content": "result1"}
        assert result[1] == {"role": "tool", "content": "result2"}

    def test_none_content_becomes_empty_string(self):
        provider = _make_provider()
        msgs = [Message(role=Role.USER, content=None)]
        result = provider.format_messages(msgs)
        assert result[0]["content"] == ""

    def test_tool_message_without_results_dropped(self):
        provider = _make_provider()
        msgs = [Message(role=Role.TOOL, tool_results=None)]
        result = provider.format_messages(msgs)
        assert result == []

    def test_message_ordering_preserved(self):
        provider = _make_provider()
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="user1"),
            Message(role=Role.ASSISTANT, content="assist1"),
            Message(role=Role.USER, content="user2"),
        ]
        result = provider.format_messages(msgs)
        assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]
        assert [m["content"] for m in result] == ["sys", "user1", "assist1", "user2"]

    def test_full_conversation_round_trip(self):
        """System → User → Assistant(tool_call) → Tool(result) → Assistant."""
        provider = _make_provider()
        msgs = [
            Message(role=Role.SYSTEM, content="You have tools"),
            Message(role=Role.USER, content="What's the weather?"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="c1", name="get_weather", arguments={"city": "NYC"})
            ]),
            Message(role=Role.TOOL, tool_results=[
                ToolResult(call_id="c1", name="get_weather", result="Sunny, 72F")
            ]),
            Message(role=Role.ASSISTANT, content="It's sunny and 72F in NYC!"),
        ]
        result = provider.format_messages(msgs)
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[3]["content"] == "Sunny, 72F"
        assert result[4]["content"] == "It's sunny and 72F in NYC!"


# ===========================================================================
# OllamaProvider.format_tools
# ===========================================================================


class TestFormatTools:
    """Tests for OllamaProvider.format_tools()."""

    def test_basic_tool(self):
        provider = _make_provider()
        tools = [_make_tool()]
        result = provider.format_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather"
        assert "properties" in result[0]["function"]["parameters"]

    def test_tool_with_no_params(self):
        provider = _make_provider()
        tools = [ToolDefinition(name="noop", description="Does nothing", parameters=[])]
        result = provider.format_tools(tools)
        assert result[0]["function"]["parameters"] == {
            "type": "object", "properties": {}, "required": []
        }

    def test_multiple_tools(self):
        provider = _make_provider()
        tools = [_make_tool("a", "A"), _make_tool("b", "B")]
        result = provider.format_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_complex_params_preserved(self):
        provider = _make_provider()
        tools = [ToolDefinition(
            name="complex",
            description="Complex tool",
            parameters=[
                ToolParameter(name="items", type="array", description="List",
                              required=True, items={"type": "string"}),
                ToolParameter(name="option", type="string", description="Opt",
                              required=False, enum=["a", "b"]),
            ]
        )]
        result = provider.format_tools(tools)
        params = result[0]["function"]["parameters"]
        assert params["properties"]["items"]["items"] == {"type": "string"}
        assert params["properties"]["option"]["enum"] == ["a", "b"]
        assert "items" in params["required"]
        assert "option" not in params["required"]


# ===========================================================================
# OllamaProvider.parse_response
# ===========================================================================


class TestParseResponse:
    """Tests for OllamaProvider.parse_response()."""

    def test_dict_response_with_content(self):
        provider = _make_provider()
        response = {
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 50,
            "eval_count": 10,
        }
        result = provider.parse_response(response)
        assert result.content == "Hello!"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.usage == {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}

    def test_dict_response_with_tool_calls(self):
        provider = _make_provider()
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "London"}}}
                ]
            },
            "done": True,
        }
        result = provider.parse_response(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "London"}
        assert result.finish_reason == "tool_calls"

    def test_arguments_as_json_string(self):
        provider = _make_provider()
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "foo", "arguments": '{"x": 42}'}}
                ]
            },
            "done": True,
        }
        result = provider.parse_response(response)
        assert result.tool_calls[0].arguments == {"x": 42}

    def test_invalid_json_args_become_empty(self):
        provider = _make_provider()
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "foo", "arguments": "not valid json"}}
                ]
            },
            "done": True,
        }
        result = provider.parse_response(response)
        assert result.tool_calls[0].arguments == {}

    def test_truncated_response(self):
        provider = _make_provider()
        response = {
            "message": {"role": "assistant", "content": "Partial..."},
            "done": False,
            "done_reason": "length",
        }
        result = provider.parse_response(response)
        assert result.finish_reason == "length"

    def test_done_false_without_reason(self):
        provider = _make_provider()
        response = {
            "message": {"role": "assistant", "content": "..."},
            "done": False,
        }
        result = provider.parse_response(response)
        assert result.finish_reason == "length"

    def test_zero_usage_returns_none(self):
        provider = _make_provider()
        response = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
            "prompt_eval_count": 0,
            "eval_count": 0,
        }
        result = provider.parse_response(response)
        assert result.usage is None

    def test_empty_name_tool_call_dropped(self):
        provider = _make_provider()
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "", "arguments": {}}},
                    {"function": {"name": "valid", "arguments": {"a": 1}}},
                ]
            },
            "done": True,
        }
        result = provider.parse_response(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "valid"

    def test_raw_content_preserved_for_tool_calls(self):
        provider = _make_provider()
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "foo", "arguments": {}}}
        ]}
        response = {"message": msg, "done": True}
        result = provider.parse_response(response)
        assert result.raw_content == msg

    def test_raw_content_none_for_text_response(self):
        provider = _make_provider()
        response = {"message": {"role": "assistant", "content": "Hello"}, "done": True}
        result = provider.parse_response(response)
        assert result.raw_content is None

    def test_content_stripped(self):
        provider = _make_provider()
        response = {"message": {"role": "assistant", "content": "  spaced  "}, "done": True}
        result = provider.parse_response(response)
        assert result.content == "spaced"

    def test_multiple_tool_calls(self):
        provider = _make_provider()
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": {"x": 1}}},
                    {"function": {"name": "b", "arguments": {"y": 2}}},
                ]
            },
            "done": True,
        }
        result = provider.parse_response(response)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_0"
        assert result.tool_calls[1].id == "call_1"


# ===========================================================================
# Error classification helpers
# ===========================================================================


class TestErrorClassification:
    """Tests for _is_retryable_error and _is_tool_unsupported_error."""

    def test_connection_refused_is_retryable(self):
        assert _is_retryable_error(Exception("Connection refused"))

    def test_timeout_is_retryable(self):
        assert _is_retryable_error(Exception("Request timeout"))

    def test_tool_unsupported_not_retryable(self):
        assert not _is_retryable_error(Exception("model does not support tools"))

    def test_random_error_not_retryable(self):
        assert not _is_retryable_error(Exception("something went wrong"))

    def test_tool_unsupported_detected(self):
        assert _is_tool_unsupported_error(
            Exception("registry.ollama.ai/library/gemma3:12b does not support tools")
        )

    def test_tool_use_not_supported(self):
        assert _is_tool_unsupported_error(Exception("tool use is not supported"))

    def test_normal_error_not_tool_unsupported(self):
        assert not _is_tool_unsupported_error(Exception("connection refused"))


# ===========================================================================
# OllamaProvider initialization
# ===========================================================================


class TestProviderInit:
    """Tests for OllamaProvider initialization and configuration."""

    def test_default_base_url(self):
        provider = _make_provider()
        assert provider._base_url == "http://localhost:11434"

    def test_custom_base_url_via_kwargs(self):
        provider = _make_provider(base_url="http://remote:11434")
        assert provider._base_url == "http://remote:11434"

    def test_env_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:11434")
        provider = OllamaProvider(api_key="", model="test")
        assert provider._base_url == "http://env-host:11434"

    def test_kwargs_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:11434")
        provider = OllamaProvider(api_key="", model="test", base_url="http://kwarg-host:11434")
        assert provider._base_url == "http://kwarg-host:11434"

    def test_provider_name(self):
        provider = _make_provider()
        assert provider.provider_name == "ollama"

    def test_model_stored(self):
        provider = OllamaProvider(api_key="", model="llama3.1")
        assert provider.model == "llama3.1"

    def test_tool_support_initially_unknown(self):
        provider = _make_provider()
        assert provider._model_supports_tools is None

    def test_client_lazy_initialized(self):
        provider = _make_provider()
        assert provider._client is None


# ===========================================================================
# OllamaProvider.complete — fallback state machine (mocked)
# ===========================================================================


class TestCompleteMethod:
    """Tests for OllamaProvider.complete() with mocked client."""

    @pytest.mark.asyncio
    async def test_no_tools_plain_completion(self):
        provider = _make_provider()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        })
        provider._client = mock_client

        result = await provider.complete(
            messages=[Message(role=Role.USER, content="Hi")],
            temperature=0.5,
        )
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        # No tools passed → no format constraint
        call_kwargs = mock_client.chat.call_args[1]
        assert "tools" not in call_kwargs
        assert "format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_native_tools_success_marks_supported(self):
        provider = _make_provider()
        provider._model_supports_tools = None  # Unknown
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "foo", "arguments": {}}}]
            },
            "done": True,
        })
        provider._client = mock_client

        result = await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            tools=[_make_tool("foo")],
        )
        assert provider._model_supports_tools is True
        assert result.tool_calls[0].name == "foo"

    @pytest.mark.asyncio
    async def test_prompt_fallback_injects_system_and_schema(self):
        """When _model_supports_tools=False, should inject prompt + use format schema."""
        provider = _make_provider()
        provider._model_supports_tools = False
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": '{"tool_calls": [{"name": "foo", "arguments": {"x": 1}}]}'},
            "done": True,
        })
        provider._client = mock_client

        tools = [_make_tool("foo")]
        result = await provider.complete(
            messages=[Message(role=Role.SYSTEM, content="Base system prompt")],
            tools=tools,
        )

        # Verify the system message was augmented
        call_kwargs = mock_client.chat.call_args[1]
        system_msg = call_kwargs["messages"][0]
        assert system_msg["role"] == "system"
        assert "foo" in system_msg["content"]  # Tool name injected
        assert "RULES" in system_msg["content"]  # Instructions injected

        # Verify format schema was set
        assert "format" in call_kwargs
        assert call_kwargs["format"]["required"] == ["tool_calls"]

        # Verify tool calls parsed from text
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "foo"
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_prompt_fallback_message_response(self):
        """Model responds with {"message": "..."} in prompt-based mode."""
        provider = _make_provider()
        provider._model_supports_tools = False
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": '{"message": "I cannot do that"}'},
            "done": True,
        })
        provider._client = mock_client

        result = await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            tools=[_make_tool()],
        )
        assert result.content == "I cannot do that"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_tool_unsupported_triggers_fallback(self):
        """Native tools fail with 'does not support tools' → recursive fallback."""
        provider = _make_provider()
        provider._model_supports_tools = None

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if "tools" in kwargs:
                raise Exception("model does not support tools")
            # Fallback call (no native tools, has format schema)
            return {
                "message": {"role": "assistant", "content": '{"tool_calls": [{"name": "foo", "arguments": {}}]}'},
                "done": True,
            }

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        provider._client = mock_client

        result = await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            tools=[_make_tool("foo")],
        )
        assert provider._model_supports_tools is False
        assert result.tool_calls[0].name == "foo"
        assert call_count == 2  # First attempt (native) + fallback

    @pytest.mark.asyncio
    async def test_retryable_error_retries(self):
        provider = _make_provider()
        mock_client = AsyncMock()

        attempts = 0

        async def failing_chat(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Exception("Connection refused")
            return {
                "message": {"role": "assistant", "content": "Finally!"},
                "done": True,
            }

        mock_client.chat = failing_chat
        provider._client = mock_client

        result = await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
        )
        assert result.content == "Finally!"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises(self):
        provider = _make_provider()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=Exception("something totally broken"))
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="Ollama API failed"):
            await provider.complete(
                messages=[Message(role=Role.USER, content="test")],
            )

    @pytest.mark.asyncio
    async def test_format_kwarg_overrides_schema(self):
        """Explicit format='json' kwarg takes priority over response schema."""
        provider = _make_provider()
        provider._model_supports_tools = False
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": '{"result": "custom"}'},
            "done": True,
        })
        provider._client = mock_client

        await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            tools=[_make_tool()],
            format="json",  # Explicit override
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["format"] == "json"  # Not the schema dict

    @pytest.mark.asyncio
    async def test_max_tokens_sets_num_predict(self):
        provider = _make_provider()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        })
        provider._client = mock_client

        await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            max_tokens=512,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["num_predict"] == 512

    @pytest.mark.asyncio
    async def test_temperature_passed(self):
        provider = _make_provider()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        })
        provider._client = mock_client

        await provider.complete(
            messages=[Message(role=Role.USER, content="test")],
            temperature=0.2,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.2
