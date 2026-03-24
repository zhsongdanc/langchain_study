from simple_agent.agent import Agent
from simple_agent.model_client import SYSTEM_PROMPT, build_model_client
from simple_agent.tools import GetCelebrityAgeTool, MultiplyTool, ToolRegistry


def main() -> None:
    tools = ToolRegistry([GetCelebrityAgeTool(), MultiplyTool()])
    model_client = build_model_client()
    agent = Agent(
        model_client=model_client,
        tool_registry=tools,
        system_prompt=SYSTEM_PROMPT,
        max_steps=5,
    )

    user_input = "周杰伦的年龄乘以 2 是多少？"
    result = agent.run(user_input)

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


if __name__ == "__main__":
    main()
