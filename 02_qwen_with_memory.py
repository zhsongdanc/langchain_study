from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 基础组件（依然是你的 Qwen）
model = ChatOllama(model="qwen2.5:3b")

# 2. 增强版 Prompt：增加了一个 "chat_history" 占位符
# 设计思想：这个占位符会自动填充之前的对话 tuple 列表
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个记性很好的编程助手。"),
    MessagesPlaceholder(variable_name="history"),  # 这里就是记忆存放的地方
    ("user", "{input}")
])

parser = StrOutputParser()


# 3. 构建基础链
chain = prompt | model | parser

# 4. 重点：给链披上一层“记忆外套” (RunnableWithMessageHistory)
# 设计思想：装饰器模式。不改动原有 chain 逻辑，只在外面套一层处理历史记录的功能。
store = {}  # 简单模拟一个数据库，用来存不同用户的对话


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 最终的可执行对象
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. 执行测试
if __name__ == "__main__":
    config = {"configurable": {"session_id": "user_01"}}  # 模拟 SessionID

    print("--- 记忆测试开始（你可以先告诉它名字，再问它是谁） ---")
    while True:
        user_input = input("我：")
        if user_input.lower() == 'exit': break

        # 注意这里调用的是封装后的 with_message_history
        response = with_message_history.invoke(
            {"input": user_input},
            config=config
        )
        print(f"AI：{response}\n")