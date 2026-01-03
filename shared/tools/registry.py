"""Tool registry for managing and executing tools."""

import logging
from typing import Dict, List, Optional, Callable, Awaitable

from shared.llm.types import ToolDefinition, ToolCall, ToolResult
from .interface import Tool, ToolContext

logger = logging.getLogger(__name__)

# Type for abort check callback
AbortCheck = Callable[[], Awaitable[Optional[str]]]


class ToolRegistry:
    """
    Central registry for all available tools.

    Manages tool registration, lookup, and execution.
    Thread-safe for concurrent tool execution.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was found and removed
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions for LLM."""
        return [tool.to_definition() for tool in self._tools.values()]

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            call: The tool call from LLM
            context: Execution context

        Returns:
            ToolResult with execution output
        """
        tool = self._tools.get(call.name)
        if not tool:
            logger.warning(f"Unknown tool requested: {call.name}")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=f"Error: Unknown tool '{call.name}'. "
                       f"Available tools: {', '.join(self._tools.keys())}",
                is_error=True
            )

        try:
            logger.info(f"Executing tool: {call.name} with args: {call.arguments}")
            result = await tool.execute(context, **call.arguments)
            logger.debug(f"Tool {call.name} result: {result[:200]}...")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=result,
                is_error=False
            )
        except TypeError as e:
            # Usually means wrong arguments
            logger.error(f"Tool {call.name} argument error: {e}")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=f"Error: Invalid arguments for {call.name}: {str(e)}",
                is_error=True
            )
        except Exception as e:
            logger.error(f"Tool {call.name} execution error: {e}", exc_info=True)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=f"Error executing {call.name}: {str(e)}",
                is_error=True
            )

    async def execute_batch(
        self,
        calls: List[ToolCall],
        context: ToolContext,
        abort_check: Optional[AbortCheck] = None
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls sequentially with optional abort checking.

        Args:
            calls: List of tool calls
            context: Execution context
            abort_check: Optional async callback that returns abort reason if should abort,
                        None to continue. Called between each tool execution.

        Returns:
            List of tool results in same order (may be shorter if aborted)
        """
        results = []
        for i, call in enumerate(calls):
            # Check for abort signal before each tool (except first)
            if i > 0 and abort_check:
                try:
                    abort_reason = await abort_check()
                    if abort_reason:
                        logger.info(f"Aborting tool batch after {i} tools: {abort_reason}")
                        # Add abort result for remaining tools
                        for remaining_call in calls[i:]:
                            results.append(ToolResult(
                                call_id=remaining_call.id,
                                name=remaining_call.name,
                                result=f"[ABORTED] Tool execution cancelled: {abort_reason}",
                                is_error=False  # Not an error, just cancelled
                            ))
                        return results
                except Exception as e:
                    logger.warning(f"Abort check failed: {e}")

            result = await self.execute(call, context)
            results.append(result)
        return results
