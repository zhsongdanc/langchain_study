from __future__ import annotations

from simple_agent.model_client import BaseModelClient
from simple_agent.schemas import AgentResult, Message, ToolCall, TraceEvent
from simple_agent.tools import ToolRegistry


class Agent:
    def __init__(
        self,
        model_client: BaseModelClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 5,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        history: list[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]
        trace: list[TraceEvent] = [
            TraceEvent(step=0, event_type="user_message", payload={"content": user_input}),
        ]

        for step in range(1, self.max_steps + 1):
            action = self.model_client.generate(history, self.tool_registry.definitions())
            trace.append(
                TraceEvent(
                    step=step,
                    event_type="model_action",
                    payload={
                        "action": action.action,
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "answer": action.answer,
                    },
                )
            )

            if action.action == "final":
                answer = action.answer or ""
                history.append(Message(role="assistant", content=answer))
                trace.append(
                    TraceEvent(
                        step=step,
                        event_type="final_answer",
                        payload={"answer": answer},
                    )
                )
                return AgentResult(answer=answer, steps=step, history=history, trace=trace)

            if action.tool_name is None:
                raise ValueError("Tool action must include tool_name.")

            trace.append(
                TraceEvent(
                    step=step,
                    event_type="tool_call",
                    payload={
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                    },
                )
            )
            history.append(
                Message(
                    role="assistant",
                    content=f"Calling tool: {action.tool_name}",
                    tool_call=ToolCall(name=action.tool_name, arguments=action.arguments),
                )
            )

            tool_result = self.tool_registry.execute(action.tool_name, action.arguments)
            trace.append(
                TraceEvent(
                    step=step,
                    event_type="tool_result",
                    payload={
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "result": tool_result,
                    },
                )
            )
            history.append(
                Message(
                    role="tool",
                    content=f"{action.tool_name}({action.arguments}) => {tool_result}",
                )
            )

        raise RuntimeError(f"Agent stopped after reaching max_steps={self.max_steps}.")
