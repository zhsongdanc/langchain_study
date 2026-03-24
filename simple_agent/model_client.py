from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import requests

from simple_agent.schemas import Message, ModelAction, ToolDefinition


SYSTEM_PROMPT = """You are a helpful assistant that can either answer directly or call a tool.

You must return valid JSON only.

Rules:
1. Do not guess missing facts.
2. If a question requires external facts or intermediate calculation, call tools step by step.
3. Do not return a final answer until the whole user request is fully solved.
4. For multi-step tasks, continue calling tools until every required step is completed.
5. If you already have one intermediate result but the user asked for more work, do not stop early.

If you need a tool, return:
{"action":"tool","tool_name":"tool_name","arguments":{"key":"value"}}

If you can answer directly, return:
{"action":"final","answer":"your answer"}

Example:
User: 周杰伦的年龄乘以 2 是多少？
First return:
{"action":"tool","tool_name":"get_celebrity_age","arguments":{"name":"周杰伦"}}

After tool result shows age is 47, do not stop. Return:
{"action":"tool","tool_name":"multiply","arguments":{"a":47,"b":2}}

After multiply returns 94, return:
{"action":"final","answer":"周杰伦的年龄乘以 2 是 94。"}
"""


class BaseModelClient(ABC):
    @abstractmethod
    def generate(self, history: list[Message], tool_definitions: list[ToolDefinition]) -> ModelAction:
        """Generate the next action from the current conversation history."""

    @staticmethod
    def parse_action(raw_text: str) -> ModelAction:
        data = json.loads(raw_text)
        return ModelAction(
            action=data["action"],
            answer=data.get("answer"),
            tool_name=data.get("tool_name"),
            arguments=data.get("arguments", {}),
        )


class OllamaModelClient(BaseModelClient):
    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, history: list[Message], tool_definitions: list[ToolDefinition]) -> ModelAction:
        prompt = self._build_prompt(history, tool_definitions)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload["response"].strip()
        return self.parse_action(raw_text)

    @staticmethod
    def _build_prompt(history: list[Message], tool_definitions: list[ToolDefinition]) -> str:
        lines = [SYSTEM_PROMPT, "", "Available tools:"]
        for tool in tool_definitions:
            lines.append(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                    ensure_ascii=False,
                )
            )

        lines.append("")
        lines.append("Conversation history:")
        for message in history:
            lines.append(f"{message.role}: {message.content}")
            if message.tool_call is not None:
                lines.append(
                    f"assistant_tool_call: "
                    f"{json.dumps({'name': message.tool_call.name, 'arguments': message.tool_call.arguments}, ensure_ascii=False)}"
                )

        lines.append("")
        lines.append("Return JSON only.")
        return "\n".join(lines)

class DemoModelClient(BaseModelClient):
    """
    A deterministic fake model client used to teach the runtime shape first.

    This lets us focus on the agent loop before wiring a real LLM.
    """

    def generate(self, history: list[Message], tool_definitions: list[ToolDefinition]) -> ModelAction:
        _ = tool_definitions
        last_message = history[-1]

        if last_message.role == "user":
            content = last_message.content
            if "年龄" in content and "乘以" in content:
                user_text = json.dumps(
                    {
                        "action": "tool",
                        "tool_name": "get_celebrity_age",
                        "arguments": {"name": "周杰伦"},
                    },
                    ensure_ascii=False,
                )
                return self.parse_action(user_text)

            final_text = json.dumps(
                {"action": "final", "answer": f"你刚才说的是：{content}"},
                ensure_ascii=False,
            )
            return self.parse_action(final_text)

        if last_message.role == "tool" and "get_celebrity_age" in last_message.content:
            age = self._extract_tool_result(last_message.content)
            tool_text = json.dumps(
                {
                    "action": "tool",
                    "tool_name": "multiply",
                    "arguments": {"a": age, "b": 2},
                },
                ensure_ascii=False,
            )
            return self.parse_action(tool_text)

        if last_message.role == "tool" and "multiply" in last_message.content:
            result = self._extract_tool_result(last_message.content)
            final_text = json.dumps(
                {"action": "final", "answer": f"周杰伦的年龄乘以 2 是 {result}。"},
                ensure_ascii=False,
            )
            return self.parse_action(final_text)

        fallback_text = json.dumps(
            {"action": "final", "answer": "我现在无法继续推进这个任务。"},
            ensure_ascii=False,
        )
        return self.parse_action(fallback_text)

    @staticmethod
    def _extract_tool_result(content: str) -> int:
        prefix, _, value = content.rpartition("=>")
        _ = prefix
        return int(value.strip())


def build_model_client() -> BaseModelClient:
    provider = os.getenv("SIMPLE_AGENT_MODEL_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        model = os.getenv("SIMPLE_AGENT_OLLAMA_MODEL", "qwen2.5-coder:3b").strip()
        base_url = os.getenv("SIMPLE_AGENT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        timeout = int(os.getenv("SIMPLE_AGENT_OLLAMA_TIMEOUT", "60"))
        return OllamaModelClient(model=model, base_url=base_url, timeout=timeout)
    return DemoModelClient()
