from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 准备“知识库”（实际开发中你会加载 PDF/TXT）
texts = [
    "Gemini 是 Google 在 2026 年发布的最新一代大模型。",
    "LangChain 的创始人是 Harrison Chase。",
    "今天的午餐菜单是红烧肉。"
]

# 2. 初始化 Embeddings 和 向量数据库
# 设计思想：Embeddings 负责把文本转换成数学坐标
embeddings = OllamaEmbeddings(model="qwen2.5:3b")
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()  # 把它变成一个“检索器”

# 3. 定义 RAG 专用 Prompt
# 设计思想：通过占位符 {context} 注入我们搜到的资料
template = """
你是一个只根据所给资料回答问题的助手。如果你不知道，就说不知道。
资料内容：{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="qwen2.5:3b")

# 4. 构建 RAG 流水线 (LCEL 的艺术)
# 注意这里的 RunnablePassthrough，它像 Java 里的“透传”，把问题传给下一级
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

# 5. 测试
if __name__ == "__main__":
    # 场景 A：问它知识库里有的
    print("回答 1：", rag_chain.invoke("谁是 LangChain 的创始人？"))

    # 场景 B：问它知识库里没有的
    print("回答 2：", rag_chain.invoke("谁是苹果公司的 CEO？"))