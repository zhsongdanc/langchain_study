# Codex Agent Loop Design Notes

这些笔记围绕 OpenAI 文章《Unrolling the Codex agent loop》展开，但重点不放在“功能介绍”，而放在更值得借鉴的运行时设计取舍上。

## 推荐阅读顺序

1. `01_state_as_append_only_log.md`
2. `02_cache_friendly_runtime.md`
3. `03_statelessness_privacy_and_compaction.md`
4. `04_portability_and_architecture_lessons.md`

## 一个总判断

如果只从“模型会调用工具，再继续推理”这个角度看，这篇文章并不算特别技术化。

但如果从系统设计角度看，它其实讲了一个更重要的问题：

如何把一个 LLM agent 做成一个能够长期运行、可追踪、可压缩、可迁移、并且成本可控的 runtime。

文章里最值得学习的不是某个单点技巧，而是下面几类设计思路：

- 用结构化 item 表示会话状态，而不是把 prompt 当成一段字符串。
- 用 append-only 的方式管理上下文，尽可能维持前缀稳定，提升缓存命中率。
- 明知可以依赖服务端会话状态，仍优先选择无状态请求，以换取更好的可移植性和隐私边界。
- 把上下文压缩当成正式的状态迁移机制，而不是临时拼接摘要。
- 把 agent runtime 与模型提供方解耦，让同一套运行框架可以落在不同 inference backend 上。

## 适合你重点看的问题

读每一篇时，可以优先带着这几个问题：

- 这个设计点解决的到底是“模型能力问题”，还是“系统工程问题”？
- 如果没有这个设计，agent 在长会话、多工具、低成本、强隐私场景下会怎样退化？
- 这个设计点是否值得在自己的 agent 框架里优先实现？
- 这里体现的是 prompt engineering，还是 runtime engineering？

## 参考

- OpenAI: Unrolling the Codex agent loop
- OpenAI API docs: Prompt caching
- OpenAI API docs: Conversation state
