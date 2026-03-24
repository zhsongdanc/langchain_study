from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
ActionType = Literal["tool", "final"]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call: ToolCall | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ModelAction:
    action: ActionType
    answer: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    steps: int
    history: list[Message]
