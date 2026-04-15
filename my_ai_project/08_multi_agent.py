'''
Description: 多智能体系统
Author: Dream Ze
Date: 2026-04-16 01:23:51
LastEditTime: 2026-04-16 01:34:18
LastEditors: Dream Ze
'''
import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ================= 1. 定义大模型 =================
llm = ChatOpenAI(model="deepseek-chat", temperature=0.7)

# ================= 2. 核心魔法：定义路由器的输出规则 =================
# 我们用 Pydantic 来规定大模型必须且只能输出什么字段
class RouteDecision(BaseModel):
    # 强制大模型只能从这三个词里选一个输出
    target_expert: Literal["coder", "musician", "general"] = Field(
        description="分析用户的输入。如果是关于代码编写、报错、程序优化，选择 'coder'；如果是关于写歌、音乐创作、寻找灵感，选择 'musician'；其他闲聊选择 'general'。"
    )

# 给大模型装上这套“紧箍咒”，它现在就是一个纯粹的分类器了！
# router_llm = llm.with_structured_output(RouteDecision)
router_llm = llm.with_structured_output(RouteDecision, method="function_calling")


# ================= 3. 定义 State 和 专家节点 =================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 专家 A：资深程序员
def coder_node(state: State):
    print("\n💻 [系统分发] -> 召唤【程序员专家】处理中...")
    # 偷偷在用户的消息前面，加上程序员的人设
    sys_msg = SystemMessage(content="你是一个顶级的 Python 架构师。说话要专业、严谨，直接给代码方案。")
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# 专家 B：词曲创作人
def musician_node(state: State):
    print("\n🎵 [系统分发] -> 召唤【音乐创作专家】处理中...")
    # 换上音乐人的人设
    sys_msg = SystemMessage(content="你是一个充满激情的流行乐词曲创作人。说话要文艺、感性，用音乐的视角启发用户，多给和弦或歌词建议。")
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# 通用接待员
def general_node(state: State):
    print("\n☕ [系统分发] -> 召唤【通用助手】处理中...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ================= 4. 定义“智能分发”逻辑 (边) =================
def route_direction(state: State) -> str:
    print("\n🔍 [路由分析] 接待员正在分析用户意图...")
    # 让带了紧箍咒的模型去读用户的最后一句话
    decision = router_llm.invoke(state["messages"])
    # 它的返回值直接就是我们定义好的 target_expert 字段！不再有任何废话！
    return decision.target_expert 


# ================= 5. 画图建构 =================
graph_builder = StateGraph[State, None, State, State](State)

# 添加三个专家节点
graph_builder.add_node("coder", coder_node)
graph_builder.add_node("musician", musician_node)
graph_builder.add_node("general", general_node)

# 关键点：起点不是连向一个节点，而是连向我们的“路由函数”
# 路由函数返回什么名字，流程就自动走到哪个节点！
graph_builder.add_conditional_edges(
    START, 
    route_direction
)

# 专家处理完后，直接结束
graph_builder.add_edge("coder", END)
graph_builder.add_edge("musician", END)
graph_builder.add_edge("general", END)

app = graph_builder.compile()

# ================= 6. 终极测试 =================
print("=== AI Partner 多智能体系统启动 ===")

# 写一个小函数来专门跑测试，让打印出来的格式更清爽
def run_test(test_name: str, query: str):
    print(f"\n[{test_name}]")
    print(f"🙋‍♂️ 你说：{query}")
    
    # 1. 运行图，并把最终返回的状态（State）存到 final_state 变量里
    final_state = app.invoke({"messages": [("user", query)]})
    
    # 2. 从最终的 State 里，拿出 messages 列表的最后一条（也就是专家的回答），打印它的内容
    final_answer = final_state["messages"][-1].content
    
    print(f"\n✨ 专家回复：\n{final_answer}")
    print("-" * 50)


# 测试 1：闲聊意图
run_test(
    "测试 1 - 闲聊意图", 
    "今天天气真好啊。"
)

# 测试 2：触发代码意图
run_test(
    "测试 2 - 代码意图", 
    "给我写一段 Python 代码，实现hello world"
)

# 测试 3：触发音乐意图
run_test(
    "测试 3 - 音乐意图", 
    "给我点灵感，我要创作一首关于失落和重新振作的歌。"
)