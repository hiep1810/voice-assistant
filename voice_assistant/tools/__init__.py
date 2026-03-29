"""Tool calling system for voice assistant."""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
import json
from datetime import datetime
import time


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None


class ToolRegistry:
    """
    Registry for tool definitions and handlers.

    Usage:
        registry = ToolRegistry()
        registry.register("get_time", "Get current time", {}, lambda: datetime.now().isoformat())
        result = registry.execute("get_time", {})
    """

    def __init__(self):
        self._tools: Dict[str, Dict] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema for parameters
            handler: Function to call when tool is invoked
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._handlers[name]

    def get_definition(self, name: str) -> Optional[Dict]:
        """Get tool definition."""
        return self._tools.get(name)

    def get_all_definitions(self) -> List[Dict]:
        """Get all tool definitions in OpenAI format."""
        return [
            {
                "type": "function",
                "function": tool,
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult with success/failure and result
        """
        if name not in self._handlers:
            return ToolResult(
                tool_name=name,
                result=None,
                success=False,
                error=f"Unknown tool: {name}",
            )

        try:
            handler = self._handlers[name]
            result = handler(**arguments)
            return ToolResult(
                tool_name=name,
                result=result,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                result=None,
                success=False,
                error=str(e),
            )


def create_builtin_tools() -> ToolRegistry:
    """Create registry with built-in tools."""
    registry = ToolRegistry()

    # Get current time
    registry.register(
        "get_current_time",
        "Get the current time in 24-hour format",
        {
            "type": "object",
            "properties": {},
            "required": [],
        },
        lambda: datetime.now().strftime("%H:%M:%S"),
    )

    # Get current date
    registry.register(
        "get_date",
        "Get the current date",
        {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Date format (default: %Y-%m-%d)",
                },
            },
            "required": [],
        },
        lambda format="%Y-%m-%d": datetime.now().strftime(format),
    )

    # Get current datetime
    registry.register(
        "get_datetime",
        "Get the current date and time",
        {
            "type": "object",
            "properties": {},
            "required": [],
        },
        lambda: datetime.now().isoformat(),
    )

    # Set timer
    registry.register(
        "set_timer",
        "Set a timer for a specified duration",
        {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Timer duration in seconds",
                },
                "label": {
                    "type": "string",
                    "description": "Optional timer label",
                },
            },
            "required": ["seconds"],
        },
        _create_timer_handler(),
    )

    # Wait/delay tool
    registry.register(
        "wait",
        "Wait for a specified duration (useful for pacing responses)",
        {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Duration to wait in seconds",
                },
            },
            "required": ["seconds"],
        },
        lambda seconds: time.sleep(seconds) or f"Waited {seconds} seconds",
    )

    return registry


def _create_timer_handler() -> Callable:
    """Create timer handler with background tracking."""
    _timers = []

    def handler(seconds: int, label: Optional[str] = None) -> str:
        end_time = time.time() + seconds
        timer_info = {
            "end_time": end_time,
            "seconds": seconds,
            "label": label or "Timer",
        }
        _timers.append(timer_info)

        # Schedule notification (simplified - just return message for now)
        return f"Timer '{label or 'Timer'}' set for {seconds} seconds"

    return handler


# Global registry instance
_builtin_registry: Optional[ToolRegistry] = None


def get_builtin_tools() -> ToolRegistry:
    """Get or create built-in tool registry."""
    global _builtin_registry
    if _builtin_registry is None:
        _builtin_registry = create_builtin_tools()
    return _builtin_registry


def reset_builtin_tools() -> None:
    """Reset built-in tools (for testing)."""
    global _builtin_registry
    _builtin_registry = None
