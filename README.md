# AI 应用开发核心笔记：从 LangChain 到 LangGraph 实战

**目标**：构建一个具备长期记忆、工具调用能力、支持多智能体路由，并能处理实时语音打断（Barge-in）的全能型 AI Partner。
**底层模型**：DeepSeek (通过兼容 OpenAI 接口接入)

***

## 第一阶段：LangChain 核心三板斧 (给大模型装上手脚)

LangChain 的核心作用是打通大模型与外部世界的连接。

### 1. 基础调用与环境解耦

- **最佳实践**：不要在代码里写死 API Key。使用 `python-dotenv` 读取 `.env` 文件。
- **DeepSeek 接入**：由于 DeepSeek 接口兼容 OpenAI，直接使用 `ChatOpenAI` 类，修改 `OPENAI_API_BASE` 即可无缝切换。

### 2. 记忆管理 (Messages)

大模型本身是“金鱼记忆”，API 每次调用都是无状态的。

- **三种消息类型**：
  - `SystemMessage`：设定人设和系统背景。
  - `HumanMessage`：用户的提问。
  - `AIMessage`：大模型的回复。
- **原理**：通过维护一个包含历史 Message 的列表（小本本），每次提问时全量发给大模型，实现多轮对话上下文。

### 3. 工具调用 (Function Calling / Tools)

让大模型具备查日历、读文件、搜索网页等执行真实任务的能力。

- **定义工具**：使用 `@tool` 装饰器包裹 Python 函数。**函数内部的注释（Docstring）极其重要**，大模型依靠注释来决定是否使用该工具。
- **绑定工具**：使用 `llm.bind_tools([工具列表])` 给模型递交“工具说明书”。
- **执行逻辑**：大模型不会直接执行代码，而是返回一个 `tool_calls` 的请求。我们需要用代码截获这个请求，真实执行 Python 函数后，将结果打包成 `ToolMessage` 重新发给大模型进行总结。

***

## 第二阶段：LangGraph 流程控制 (干掉 `if-else` 的流程经理)

当 AI 逻辑变复杂（调用多个工具、循环重试、多专家协作）时，LangChain 就变得难以维护。LangGraph 引入了“状态机”和“图”的思维。

### 1. LangGraph 核心三要素

- **State (全局状态)**：图里所有节点共享的数据结构（通常是一个包含 messages 列表的字典）。相当于整个流程流转中传递的“小本本”。
- **Node (节点)**：执行具体任务的单元。比如“大模型思考节点”、“工具执行节点”。
- **Edge (边)**：决定流程走向。最强大的是**条件边（Conditional Edge）**，例如 `tools_condition`，能自动根据大模型是否请求工具来决定下一站去哪。

### 2. 长期记忆与持久化 (Checkpointer)

让 AI Partner 拥有真正的“记忆”，能够记住昨天的对话。

- **机制**：通过在编译图时传入 `checkpointer`，并指定 `thread_id`。
- **内存存储**：`MemorySaver`（用于测试，重启丢失）。
- **数据库存储**：`SqliteSaver` 或 `PostgresSaver`（用于生产，数据落地到 `.db` 物理文件，宕机不丢失）。

### 3. 终极大招：随时打断 (Update State)

这是实时语音通讯 (RTC) 场景下的核心痛点解决方案。

- **场景**：当 AI 正在流式输出或播报语音时，用户突然说话打断。
- **原理**：利用 `app.update_state()` 这个“上帝之手”。强制停止当前节点的输出，并将用户插嘴的最新指令，强行写入当前 `thread_id` 对应的 State 中。再次触发大模型时，它就能顺畅接上被打断的话茬。

### 4. 多智能体路由 (Multi-Agent Routing)

让系统根据用户意图，自动切换“人格”和“专家”（例如在严谨的程序员和感性的音乐创作人之间切换）。

- **核心技术**：`with_structured_output` 强制大模型输出规定好的 JSON/Pydantic 结构。
- **DeepSeek 兼容性避坑**：DeepSeek 目前未完全支持最新 JSON Schema，必须在代码中明确指定 `method="function_calling"` 才能正常工作。
- **架构实现**：
  1. 定义一个 Router LLM，仅用于分析意图。
  2. 在 `START` 节点连接条件边，调用 Router。
  3. 根据 Router 的返回值，将流程导向不同的专家 Node（Coder 节点 / Musician 节点 / 全能助理节点）。

***

## 终极系统架构图

构建企业级 AI Partner 的终极形态：

```text
       📂【持久化数据库 (SQLite)】 
       (根据用户的 thread_id 拉取/保存状态)
                 |
             [ START ]
                 |
         🔍 Router (意图路由)
                 |
        +--------+--------+
        |                 |
  💻 Coder Node    ☕ Assistant Node (绑定了工具)
  (专注写代码)            |
        |                 +---> 🔀 tools_condition
        |                 |           |
        |                 +<---- 🛠️ Tool Node
        |                             |
        +--------+--------+-----------+
                 |
              [ END ]
                 |
       💾【状态自动写回数据库】
```

