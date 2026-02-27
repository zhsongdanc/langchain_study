from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 第一步：初始化【模型 (Model)】 ---
# 设计思想：模型是可插拔的。今天用 Qwen，明天换 GPT 只需要改这行。
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7  # 创造力设置：0为严谨，1为狂野
)



# --- 第二步：定义【提示词模版 (Prompt)】 ---
# 设计思想：不要硬编码字符串！通过模版化，实现逻辑与内容的分离。
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位幽默的编程老师，喜欢用生活中的例子解释技术名词。"),
    ("user", "请帮我解释一下什么是 {concept}")
])

# --- 第三步：定义【输出解析器 (Output Parser)】 ---
# 设计思想：模型默认输出的是复杂对象，Parser 负责把其中的文字“抠”出来。
parser = StrOutputParser()

# --- 第四步：构建【流水线 (Chain)】 ---
# 设计思想：这就是 LCEL 的精髓！用 | 符号把组件串联。
# 数据流向：用户输入 -> Prompt填充 -> Model计算 -> Parser清洗
chain = prompt | model | parser

# --- 第五步：执行 ---
if __name__ == "__main__":
    # 调用 invoke 方法，传入模版所需的变量
    target_concept = "LangChain 的链式设计"

    print(f"--- 正在向 Qwen 请求关于 '{target_concept}' 的解释 ---\n")

    # 这一行触发了整条流水线
    response = chain.invoke({"concept": target_concept})

    print(response)