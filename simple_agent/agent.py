from __future__ import annotations

from typing import Callable, Union
from uuid import uuid4

from simple_agent.compactor import SimpleCompactor
from simple_agent.model_client import BaseModelClient
from simple_agent.schemas import (
    AgentResult,
    Message,
    SuspendedRun,
    ToolCall,
    TraceEvent,
    WorkflowGraph,
    WorkflowState,
)
from simple_agent.tools import ToolRegistry


NodeHandler = Callable[[WorkflowState], Union[WorkflowState, AgentResult, SuspendedRun]]


class Agent:
    def __init__(
        self,
        model_client: BaseModelClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        compactor: SimpleCompactor | None = None,
        tools_requiring_approval: set[str] | None = None,
        max_steps: int = 5,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.compactor = compactor or SimpleCompactor()
        self.tools_requiring_approval = tools_requiring_approval or set()
        self.max_steps = max_steps
        self.graph = self._build_graph()

    def run(self, user_input: str) -> AgentResult | SuspendedRun:
        state = self._build_initial_state(user_input)
        return self._run_graph(state, self.graph)

    def resume(self, suspended_run: SuspendedRun, approved: bool) -> AgentResult | SuspendedRun:
        state = suspended_run.state
        if state.pending_approval_id != suspended_run.approval_id:
            raise ValueError("Approval id does not match suspended state.")

        state.approval_granted = approved
        state.waiting_for_approval = False
        state.trace.append(
            TraceEvent(
                step=state.step,
                event_type="approval_result",
                payload={
                    "approval_id": suspended_run.approval_id,
                    "tool_name": suspended_run.requested_tool_name,
                    "approved": approved,
                },
            )
        )
        return self._run_graph(state, self.graph, start_node="approval_result")

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
        if action.tool_name in self.tools_requiring_approval:
            return "approval"
        return "execute_tool"

    def _route_next_node(self, current_node: str, state: WorkflowState) -> str:
        if current_node == "decide":
            return self._route_after_decide(state)
        if current_node == "approval":
            raise ValueError("Approval node should suspend instead of routing immediately.")
        if current_node == "approval_result":
            if state.approval_granted:
                return "execute_tool"
            return "finish"
        if current_node == "execute_tool":
            return "decide"
        raise ValueError(f"Node {current_node} does not have a next route.")

    def _build_graph(self) -> WorkflowGraph:
        node_registry: dict[str, NodeHandler] = {
            "decide": self._decide_step,
            "approval": self._approval_step,
            "approval_result": self._approval_result_step,
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

    def _approval_step(self, state: WorkflowState) -> SuspendedRun:
        action = state.current_action
        if action is None or action.tool_name is None:
            raise ValueError("Approval step requires a pending tool action.")

        approval_id = str(uuid4())
        state.waiting_for_approval = True
        state.pending_approval_id = approval_id
        state.trace.append(
            TraceEvent(
                step=state.step,
                event_type="approval_requested",
                payload={
                    "approval_id": approval_id,
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                },
            )
        )
        return SuspendedRun(
            approval_id=approval_id,
            state=state,
            requested_tool_name=action.tool_name,
            requested_arguments=action.arguments,
        )

    def _approval_result_step(self, state: WorkflowState) -> WorkflowState:
        if state.approval_granted is None:
            raise ValueError("Approval result step requires an approval decision.")
        state.pending_approval_id = None
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

    def _run_graph(
        self,
        state: WorkflowState,
        graph: WorkflowGraph,
        start_node: str | None = None,
    ) -> AgentResult | SuspendedRun:
        current_node = start_node or graph.start_node

        while True:
            handler = graph.node_registry.get(current_node)
            if handler is None:
                raise ValueError(f"Unknown workflow node: {current_node}")

            result = handler(state)
            if isinstance(result, AgentResult):
                return result
            if isinstance(result, SuspendedRun):
                return result

            state = result
            current_node = graph.router(current_node, state)
