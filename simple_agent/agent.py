from __future__ import annotations

from typing import Callable, Union

from simple_agent.compactor import SimpleCompactor
from simple_agent.model_client import BaseModelClient
from simple_agent.schemas import AgentResult, Message, ToolCall, TraceEvent, WorkflowGraph, WorkflowState
from simple_agent.tools import ToolRegistry


NodeHandler = Callable[[WorkflowState], Union[WorkflowState, AgentResult]]


class Agent:
    def __init__(
        self,
        model_client: BaseModelClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        compactor: SimpleCompactor | None = None,
        max_steps: int = 5,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.compactor = compactor or SimpleCompactor()
        self.max_steps = max_steps
        self.graph = self._build_graph()

    def run(self, user_input: str) -> AgentResult:
        state = self._build_initial_state(user_input)
        return self._run_graph(state, self.graph)

    def _build_initial_state(self, user_input: str) -> WorkflowState:
        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]
        trace = [TraceEvent(step=0, event_type="user_message", payload={"content": user_input})]
        return WorkflowState(history=history, trace=trace)

    def _decide_step(self, state: WorkflowState) -> WorkflowState:
        if state.step >= self.max_steps:
            raise RuntimeError(f"Agent stopped after reaching max_steps={self.max_steps}.")
        next_step = state.step + 1
        action = self.model_client.generate(state.history, self.tool_registry.definitions())
        state.step = next_step
        state.current_action = action
        state.trace.append(
            TraceEvent(
                step=next_step,
                event_type="model_action",
                payload={
                    "action": action.action,
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "answer": action.answer,
                },
            )
        )
        return state

    def _route_after_decide(self, state: WorkflowState) -> str:
        action = state.current_action
        if action is None:
            raise ValueError("Workflow routing requires current_action.")
        if action.action == "final":
            return "finish"
        return "execute_tool"

    def _route_next_node(self, current_node: str, state: WorkflowState) -> str:
        if current_node == "decide":
            return self._route_after_decide(state)
        if current_node == "execute_tool":
            return "decide"
        raise ValueError(f"Node {current_node} does not have a next route.")

    def _build_graph(self) -> WorkflowGraph:
        node_registry: dict[str, NodeHandler] = {
            "decide": self._decide_step,
            "execute_tool": self._execute_tool_step,
            "finish": self._finish_step,
        }
        return WorkflowGraph(
            start_node="decide",
            node_registry=node_registry,
            router=self._route_next_node,
        )

    def _execute_tool_step(self, state: WorkflowState) -> WorkflowState:
        action = state.current_action
        if action is None or action.tool_name is None:
            raise ValueError("Tool execution requires a tool action.")

        state.trace.append(
            TraceEvent(
                step=state.step,
                event_type="tool_call",
                payload={
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                },
            )
        )
        state.history.append(
            Message(
                role="assistant",
                content=f"Calling tool: {action.tool_name}",
                tool_call=ToolCall(name=action.tool_name, arguments=action.arguments),
            )
        )

        tool_result = self.tool_registry.execute(action.tool_name, action.arguments)
        state.trace.append(
            TraceEvent(
                step=state.step,
                event_type="tool_result",
                payload={
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "result": tool_result,
                },
            )
        )
        state.history.append(
            Message(
                role="tool",
                content=f"{action.tool_name}({action.arguments}) => {tool_result}",
            )
        )
        return state

    def _finish_step(self, state: WorkflowState) -> AgentResult:
        action = state.current_action
        if action is None:
            raise ValueError("Finish step requires current_action.")

        answer = action.answer or ""
        state.final_answer = answer
        state.history.append(Message(role="assistant", content=answer))
        state.trace.append(
            TraceEvent(
                step=state.step,
                event_type="final_answer",
                payload={"answer": answer},
            )
        )
        compacted_history = self.compactor.compact(state.history)
        return AgentResult(
            answer=answer,
            steps=state.step,
            history=state.history,
            trace=state.trace,
            compacted_history=compacted_history,
        )

    def _run_graph(self, state: WorkflowState, graph: WorkflowGraph) -> AgentResult:
        current_node = graph.start_node

        while True:
            handler = graph.node_registry.get(current_node)
            if handler is None:
                raise ValueError(f"Unknown workflow node: {current_node}")

            result = handler(state)
            if isinstance(result, AgentResult):
                return result

            state = result
            current_node = graph.router(current_node, state)
