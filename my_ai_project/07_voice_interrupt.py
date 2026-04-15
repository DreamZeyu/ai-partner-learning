'''
Description: 模拟语音打断场景
Author: Dream Ze
Date: 2026-04-16 01:05:30
LastEditTime: 2026-04-16 01:05:37
LastEditors: Dream Ze
'''
import time
import threading
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ================= 1. 基础图配置 =================
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

def chatbot(state: State):
    # 这里我们不用 invoke，改用 stream 模拟大模型一个字一个字往外吐
    print("\n🎧 [AI 开始说话]: ", end="", flush=True)
    response_content = ""
    
    # 模拟流式输出
    for chunk in llm.stream(state["messages"]):
        # 假设这里有一个全局变量 interrupted 标志位
        if globals().get("is_interrupted", False):
            print("\n\n💥 [系统拦截]: 检测到用户说话！大模型被迫闭嘴。")
            # 被打断时，我们把“说到一半的话”返回，并加上一个特殊标记
            return {"messages": [("ai", response_content + "...(被打断)")]}
            
        print(chunk.content, end="", flush=True)
        response_content += chunk.content
        time.sleep(0.05) # 模拟语音合成的延迟
        
    print("\n")
    return {"messages": [("ai", response_content)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# ================= 2. 模拟真实 RTC 打断场景 =================
print("=== 火山引擎 AI 实时通话模拟 (打断测试) ===")
config = {"configurable": {"thread_id": "rtc_session_001"}}
is_interrupted = False

# 1. 正常开始第一轮
print("\n--- 第一回合：用户提问 ---")
input1 = {"messages": [("user", "请给我讲一个很长很长的孙悟空的故事。")]}
app.invoke(input1, config=config)


# 2. 第二轮：我们用多线程模拟“AI说话时，用户突然插嘴”
print("\n--- 第二回合：用户打断测试 ---")
input2 = {"messages": [("user", "故事里的猪八戒在干嘛？")]}

# 开启一个后台线程运行大模型
is_interrupted = False
ai_thread = threading.Thread(target=app.invoke, args=(input2, config))
ai_thread.start()

# 主线程等待 1.5 秒后，用户突然开口说话（触发 VAD）
time.sleep(1.5)
print("\n\n🎤 [VAD 触发]: 用户突然开口说话: '别讲猪八戒了，我想听哪吒的！'")
is_interrupted = True # 发送打断信号给大模型

# 等待刚才被打断的线程结束
ai_thread.join() 

# ================= 3. 核心：上帝之手 update_state =================
# 此时 AI 虽然闭嘴了，但我们要把用户的新指令强行塞进记忆里！
print("\n--- 第三回合：系统重整态势，快速响应 ---")

# 使用 update_state 强行把用户的最新语音识别文字，写进历史记录
app.update_state(
    config,
    {"messages": [("user", "别讲猪八戒了，立刻给我讲两句哪吒的故事！")]}
)

# 再次触发大模型生成
print("🔄 [系统]: 带着打断记忆，重新让 AI 响应...")
final_state = app.invoke(None, config=config) # 传入 None，它会自动根据最新的 State 往下跑