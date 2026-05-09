from simple_agent.agent import Agent
from simple_agent.compactor import SimpleCompactor
from simple_agent.model_client import SYSTEM_PROMPT, build_model_client
from simple_agent.schemas import AgentResult, SuspendedRun
from simple_agent.tools import GetCelebrityAgeTool, MultiplyTool, ToolRegistry


def main() -> None:
    tools = ToolRegistry([GetCelebrityAgeTool(), MultiplyTool()])
    model_client = build_model_client()
    compactor = SimpleCompactor()

    agent = Agent(
        model_client=model_client,
        tool_registry=tools,
        system_prompt=SYSTEM_PROMPT,
        compactor=compactor,
        tools_requiring_approval={"multiply"},
        max_steps=5,
    )

    user_input = "周杰伦的年龄乘以 2 是多少？"
    run_result = agent.run(user_input)
    if isinstance(run_result, SuspendedRun):
        print(
            f"[suspended] approval_id={run_result.approval_id} "
            f"tool={run_result.requested_tool_name} arguments={run_result.requested_arguments}"
        )
        result = agent.resume(run_result, approved=True)
    else:
        result = run_result

    if not isinstance(result, AgentResult):
        raise RuntimeError("Expected final AgentResult after resume.")

    print(f"Question: {user_input}")
    print(f"Model client: {model_client.__class__.__name__}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.steps}")
    print("\nHistory:")
    for message in result.history:
        print(f"- {message.role}: {message.content}")

    print("\nTrace:")
    for event in result.trace:
        print(f"- step={event.step} type={event.event_type} payload={event.payload}")

    print("\nCompacted History:")
    for message in result.compacted_history:
        print(f"- {message.role}: {message.content}")


if __name__ == "__main__":
    main()
