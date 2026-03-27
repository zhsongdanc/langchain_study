from __future__ import annotations

from simple_agent.schemas import Message


class SimpleCompactor:
    """
    A rule-based compactor that keeps user-facing conversation
    and rewrites known tool outputs into compact facts.
    """

    def compact(self, history: list[Message]) -> list[Message]:
        compacted: list[Message] = []

        for message in history:
            if message.role in {"system", "user"}:
                compacted.append(message)
                continue

            if message.role == "assistant" and message.tool_call is None:
                compacted.append(message)
                continue

            if message.role == "tool":
                fact_message = self._tool_result_to_fact(message.content)
                if fact_message is not None:
                    compacted.append(fact_message)

        return compacted

    @staticmethod
    def _tool_result_to_fact(content: str) -> Message | None:
        if content.startswith("get_celebrity_age("):
            if "周杰伦" in content:
                result = content.rsplit("=>", 1)[-1].strip()
                return Message(role="fact", content=f"Fact: 周杰伦的年龄是 {result}。")

        if content.startswith("multiply("):
            result = content.rsplit("=>", 1)[-1].strip()
            return Message(role="fact", content=f"Fact: 最近一次乘法计算结果是 {result}。")

        return None
