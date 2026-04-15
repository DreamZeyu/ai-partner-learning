import os
import sqlite3
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ================= 1. 定义底层模型与工具 =================
llm = ChatOpenAI(model="deepseek-chat", temperature=0.7)

@tool
def check_schedule(date: str) -> str:
    """查询指定日期的日程安排。参数 date 是日期字符串，如 '今天'。"""
    print(f"\n   ⚙️ [工具调用] 正在查询数据库: {date} 的安排...")
    if "今天" in date:
        return "下午3点和老板开会，讨论 AI Partner 架构方案。"
    return "没有安排"

# ================= 2. 定义路由与意图识别 =================
class RouteDecision(BaseModel):
    target_expert: Literal["coder", "assistant"] = Field(
        description="如果是关于代码、bug、编程的，选择 'coder'；如果是闲聊、查日程、其他问题，选择 'assistant'。"
    )

# 强制模型输出路由结果 (记得加上 method="function_calling" 解决格式兼容问题)
router_llm = llm.with_structured_output(RouteDecision, method="function_calling")


# ================= 3. 定义全局状态与专家节点 =================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 专家 A：程序员（不带工具，纯输出代码）
def coder_node(state: State):
    print("\n💻 [系统分发] -> 召唤【程序员专家】...")
    sys_msg = SystemMessage(content="你是资深程序员，请用最精简的代码回答问题。")
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# 专家 B：全能助理（带工具！这就是我们把前面的知识套娃进来的地方）
llm_with_tools = llm.bind_tools([check_schedule])

def assistant_node(state: State):
    print("\n☕ [系统分发] -> 召唤【全能助理】...")
    sys_msg = SystemMessage(content="你是贴心的生活助理，可以通过工具查询日程，说话要温柔。")
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


# ================= 4. 画图：上帝视角的连线 =================
graph_builder = StateGraph(State)

# 放入所有节点
graph_builder.add_node("coder", coder_node)
graph_builder.add_node("assistant", assistant_node)
graph_builder.add_node("tools", ToolNode(tools=[check_schedule]))

# 1. 起点的路由连线
def route_direction(state: State) -> str:
    decision = router_llm.invoke(state["messages"])
    return decision.target_expert

graph_builder.add_conditional_edges(START, route_direction)

# 2. 程序员走完直接结束
graph_builder.add_edge("coder", END)

# 3. 全能助理走完后，判断要不要用工具
graph_builder.add_conditional_edges("assistant", tools_condition)
# 如果用了工具，工具跑完要回传给全能助理继续说
graph_builder.add_edge("tools", "assistant")


# ================= 5. 装载数据库硬盘 =================
conn = sqlite3.connect("ultimate_ai.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph_builder.compile(checkpointer=memory)


# ================= 6. 终极测试启动 =================
print("=== 🚀 Ultimate AI Partner 引擎启动 ===")
config = {"configurable": {"thread_id": "vip_user_001"}}

def chat_with_system(query: str):
    print(f"\n🙋‍♂️ 你说：{query}")
    final_state = app.invoke({"messages": [("user", query)]}, config=config)
    print(f"\n✨ 最终回复：{final_state['messages'][-1].content}")
    print("-" * 50)

# 测试一：纯闲聊 + 注入身份
chat_with_system("你好，我是阿强，我是一名后端工程师。")

# 测试二：触发工具调用
chat_with_system("我今天下午有什么安排吗？")

# 测试三：触发跨节点路由
chat_with_system("你能帮我写一个 Python 的快速排序算法吗？")

# 测试四：终极记忆测试（跨节点是否还能记住身份）
chat_with_system("刚才那个算法写得不错！对了，你还记得我叫什么名字、做什么工作的吗？")