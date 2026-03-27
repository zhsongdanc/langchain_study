from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool", "fact"]
ActionType = Literal["tool", "final"]
TraceEventType = Literal[
    "user_message",
    "model_action",
    "tool_call",
    "tool_result",
    "final_answer",
]


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
class TraceEvent:
    step: int
    event_type: TraceEventType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    steps: int
    history: list[Message]
    trace: list[TraceEvent] = field(default_factory=list)
    compacted_history: list[Message] = field(default_factory=list)
