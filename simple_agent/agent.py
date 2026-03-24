from __future__ import annotations

from simple_agent.model_client import BaseModelClient
from simple_agent.tools import ToolRegistry
from simple_agent.schemas import AgentResult, Message, ToolCall


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

        for step in range(1, self.max_steps + 1):
            action = self.model_client.generate(history, self.tool_registry.definitions())

            if action.action == "final":
                answer = action.answer or ""
                history.append(Message(role="assistant", content=answer))
                return AgentResult(answer=answer, steps=step, history=history)

            if action.tool_name is None:
                raise ValueError("Tool action must include tool_name.")

            history.append(
                Message(
                    role="assistant",
                    content=f"Calling tool: {action.tool_name}",
                    tool_call=ToolCall(name=action.tool_name, arguments=action.arguments),
                )
            )

            tool_result = self.tool_registry.execute(action.tool_name, action.arguments)
            history.append(
                Message(
                    role="tool",
                    content=f"{action.tool_name}({action.arguments}) => {tool_result}",
                )
            )

        raise RuntimeError(f"Agent stopped after reaching max_steps={self.max_steps}.")
