# Simple Agent

这是一个用于学习 agent runtime 核心结构的最小实现。

## 当前版本包含什么

- `Agent` 负责主循环
- `ModelClient` 负责返回下一步动作
- `Tool` 使用对象式抽象
- `ToolRegistry` 负责管理和执行工具
- `Message` 作为历史记录项
- `JSON` 作为 agent 和 model 之间的简化协议
- 支持 `DemoModelClient` 和 `OllamaModelClient`
- 内置最小 `execution trace / event log`

## 目录说明

- `schemas.py`: 基础数据结构
- `tools.py`: Tool 抽象和注册中心
- `model_client.py`: 模型客户端抽象，包含演示模型和 Ollama 实现
- `agent.py`: Agent 主循环
- `main.py`: 启动入口

## 当前主流程

1. 用户输入问题
2. `Agent` 把历史消息传给 `ModelClient`
3. `ModelClient` 返回一个动作
4. 如果动作是 `tool`，则执行工具并把结果写回历史
5. 如果动作是 `final`，则结束

## Trace 设计

当前版本已经记录最小事件流，每条事件包含：

- `step`
- `event_type`
- `payload`

第一版支持的事件类型有：

- `user_message`
- `model_action`
- `tool_call`
- `tool_result`
- `final_answer`

这一步的目标不是做复杂可视化，而是先把 agent 的执行轨迹显式化，为后面的上下文管理和 workflow 做铺垫。

## 运行方式

```bash
python3 -m simple_agent.main
```

## 模型切换

默认使用 `DemoModelClient`。

如果要切到本地 Ollama：

```bash
export SIMPLE_AGENT_MODEL_PROVIDER=ollama
export SIMPLE_AGENT_OLLAMA_MODEL=qwen2.5-coder:3b
python3 -m simple_agent.main
```

也可以自定义地址和超时：

```bash
export SIMPLE_AGENT_OLLAMA_BASE_URL=http://127.0.0.1:11434
export SIMPLE_AGENT_OLLAMA_TIMEOUT=60
```

## 为什么同时保留假模型

第一版先用 `DemoModelClient`，是为了把注意力放在 runtime 架构上，而不是 API 细节。

等这一版结构看顺了，再切到 `OllamaModelClient`，你会更容易理解“相同 runtime，不同 model adapter”是什么意思。
