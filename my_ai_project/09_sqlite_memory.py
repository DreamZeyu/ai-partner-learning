'''
Description: 数据库持久化记忆
Author: Dream Ze
Date: 2026-04-16 01:39:30
LastEditTime: 2026-04-16 01:39:39
LastEditors: Dream Ze
'''
import os
import sqlite3
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# 🌟 核心变化：引入 Sqlite 存档器
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ================= 1. 基础节点和图配置 =================
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="deepseek-chat", temperature=0.7)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# ================= 2. 核心：挂载本地数据库 =================
# 连接（或自动创建）一个本地的数据库文件，名字叫 ai_partner_history.db
# check_same_thread=False 是为了避免多线程调用时报错
conn = sqlite3.connect("ai_partner_history.db", check_same_thread=False)

# 用这个数据库连接，实例化一个真实的硬盘存档器
memory = SqliteSaver(conn)

# 把硬盘装进图里
app = graph_builder.compile(checkpointer=memory)


# ================= 3. 运行测试 =================
print("=== 💾 AI Partner 数据库持久化启动 ===")
print("提示：你可以随时输入 '退出' 来关闭程序。\n")

# 我们的抽屉编号还是这个
config = {"configurable": {"thread_id": "user_vip_888"}}

while True:
    user_input = input("🙋‍♂️ 你说：")
    if user_input in ["退出", "quit", "exit"]:
        print("系统已关闭。你的记忆已被永久封存在 ai_partner_history.db 中！")
        break
        
    # 我们用 app.stream 来流转图，并抓取 chatbot 节点的输出
    for event in app.stream({"messages": [("user", user_input)]}, config=config):
        for node_name, node_state in event.items():
            if node_name == "chatbot":
                print(f"🤖 DeepSeek：{node_state['messages'][-1].content}\n")