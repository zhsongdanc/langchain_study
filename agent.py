from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


# 1. 定义工具 (Tools)
# 使用 @tool 装饰器，LangChain 会通过反射读取函数的 docstring 传给 AI
@tool
def get_celebrity_age(name: str) -> int:
    """当你需要查询名人的真实年龄时调用此工具。"""
    # 模拟搜索结果：实际开发中这里会调搜索引擎 API
    data = {"周杰伦": 47, "马斯克": 54, "雷军": 56}
    print(f"--- [工具调用] 正在查询 {name} 的年龄... ---")
    return data.get(name, 30)


@tool
def multiply(a: float, b: float) -> float:
    """当你需要计算两个数字相乘时调用此工具。"""
    print(f"--- [工具调用] 正在计算 {a} * {b}... ---")
    return a * b


tools = [get_celebrity_age, multiply]

# 2. 初始化模型 (注意：模型必须支持 Tool Calling)
# Qwen2.5 3B/7B 完美支持这个特性
model = ChatOllama(model="qwen2.5:3b")

# 3. 定义 Prompt (Agent 专用模版)
# 这里必须包含 agent_scratchpad 占位符，用来存放 AI 的思考过程
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个全能助手，可以调用工具来回答问题。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. 构造 Agent (大管家)
# 设计思想：把模型、工具、提示词绑在一起
agent = create_tool_calling_agent(model, tools, prompt)

# 5. 构造执行器 (AgentExecutor 相当于 Java 里的线程池或运行容器)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 执行任务
if __name__ == "__main__":
    query = "周杰伦的年龄乘以 2 是多少？"
    print(f"任务开始：{query}\n")

    # AI 会自主决定：先查年龄，再算乘法
    result = agent_executor.invoke({"input": query})
    print(f"\n最终结论：{result['output']}")