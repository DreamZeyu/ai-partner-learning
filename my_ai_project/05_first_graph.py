import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 🌟 新增：引入 LangGraph 自带的内存存档器
from langgraph.checkpoint.memory import MemorySaver 

load_dotenv()

# ================= 1. 定义工具和模型 =================
@tool
def check_schedule(date: str) -> str:
    """查询指定日期的日程安排。参数 date 是日期字符串，如 '今天'。"""
    print(f"\n   ⚙️ [生命周期 - 工具执行] 正在查询 {date} 的数据库...")
    if "今天" in date:
        return "上午10点开会，下午2点优化 AI Partner 的代码结构"
    return "没有安排"

llm = ChatOpenAI(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools([check_schedule])

# ================= 2. 定义全局状态 =================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ================= 3. 定义节点 =================
def chatbot(state: State):
    print("\n🟢 [生命周期 - 节点执行] chatbot 节点开始思考...")
    print(f"   📊 [状态更新] 当前小本本里共有 {len(state['messages'])} 条消息")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ================= 4. 画图并加入“存档器” =================
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[check_schedule]))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# 🌟 核心改变 1：实例化一个硬盘（存档器）
memory = MemorySaver()

# 🌟 核心改变 2：在编译打包的时候，把硬盘装上去！
app = graph_builder.compile(checkpointer=memory)

# ================= 5. 跑起来试试：多轮对话测试 =================
print("=== AI Partner 系统启动 (带长期存储) ===")

# 🌟 核心改变 3：定义一个 config，指定当前是哪个用户的哪个对话框
# 如果你要做复杂的应用，不同的用户就可以有不同的 thread_id
config = {"configurable": {"thread_id": "user_x_session_001"}}

print("\n--- 第一轮对话 ---")
input1 = {"messages": [("user", "我今天有什么安排？")]}

# 运行时带上 config
for event in app.stream(input1, config=config):
    for node_name, node_state in event.items():
        print(f"🔵 [生命周期 - 节点完成] {node_name} 节点流转完毕！")

# 现在的 get_state 绝对不会报错了！因为它能去硬盘里根据 thread_id 把记录翻出来
current_state = app.get_state(config)
print(f"\n✅ 第一轮最终回复: {current_state.values['messages'][-1].content}")


print("\n\n--- 第二轮对话（不传历史记录，看看图能不能自己想起来） ---")
# 注意：我们这里只传了新的一句话，之前的记录全靠图里的 Checkpointer 自动管理
input2 = {"messages": [("user", "那我下午的具体任务是什么？")]}

for event in app.stream(input2, config=config):
    pass # 这里为了清爽我们省略了内部生命周期的打印

final_state = app.get_state(config)
print(f"\n✅ 第二轮最终回复: {final_state.values['messages'][-1].content}")