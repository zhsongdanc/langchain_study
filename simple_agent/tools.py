from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from simple_agent.schemas import ToolDefinition


class BaseTool(ABC):
    """Object-oriented tool contract that keeps execution separate from orchestration."""

    name: str
    description: str
    parameters: dict[str, Any]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Any:
        """Run the tool with validated arguments."""


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool.execute(arguments)


class GetCelebrityAgeTool(BaseTool):
    name = "get_celebrity_age"
    description = "Get the age of a celebrity by name."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Celebrity name"},
        },
        "required": ["name"],
    }

    def execute(self, arguments: dict[str, Any]) -> Any:
        name = arguments["name"]
        data = {"周杰伦": 47, "马斯克": 54, "雷军": 56}
        return data.get(name, 30)


class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    }

    def execute(self, arguments: dict[str, Any]) -> Any:
        return arguments["a"] * arguments["b"]
