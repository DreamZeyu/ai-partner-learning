import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import messages_to_dict
# 注意这里多引入了一个 ToolMessage，专门用来装工具执行的结果
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# 1. 准备工具
@tool
def check_schedule(date: str) -> str:
    """查询指定日期的日程安排。参数 date 是日期字符串，如 '今天'。"""
    print(f"\n⚙️ [系统后台] 正在执行 Python 函数，查询 {date} 的数据库...\n")
    if "今天" in date:
        return "上午10点开会，下午2点优化 AI Partner 的 Python 代码"
    return "没有安排"

tools = [check_schedule]
llm_with_tools = llm.bind_tools(tools)

# 2. 准备聊天记录小本本
messages = [HumanMessage(content="我今天有什么安排？")]

print("你：我今天有什么安排？")

# ----------------- 第一回合：大模型思考并决定使用工具 -----------------
print("\n--- 第一回合：大模型思考 ---")
ai_msg = llm_with_tools.invoke(messages)
# 把大模型的回复（也就是那张“请求调用工具的小纸条”）记在小本本上
messages.append(ai_msg)
readable_messages = messages_to_dict(messages)
print('messages的数据结构内容:')
print(json.dumps(readable_messages, ensure_ascii=False, indent=2))
# 检查大模型是不是真的打算用工具
if ai_msg.tool_calls:
    print(f"大模型决定使用工具：{ai_msg.tool_calls[0]['name']}")
    
    # ----------------- 第二回合：代码自动执行工具 -----------------
    print("\n--- 第二回合：代码自动执行工具 ---")
    for tool_call in ai_msg.tool_calls:
        # 提取大模型给的参数
        args = tool_call["args"] 
        print('args意思',args)
        # 调用我们自己写的 Python 函数，拿到真实结果
        tool_result = check_schedule.invoke(args) 
        
        print(f"工具返回的真实结果：{tool_result}")
        
        # ⚠️ 关键点：把真实结果打包成 ToolMessage，塞进小本本
        # 必须带上 tool_call_id，这样大模型才知道这个结果对应它刚才的哪个请求
        tool_msg = ToolMessage(
            content=tool_result, 
            tool_call_id=tool_call["id"]
        )
        messages.append(tool_msg)
        
    # ----------------- 第三回合：把结果交还给大模型，让它好好说话 -----------------
    print("\n--- 第三回合：大模型整合结果并回复 ---")
    # 再次把更新后的小本本（包含了人类提问、大模型请求、工具结果）发给大模型
    final_response = llm_with_tools.invoke(messages)
    print(f"\nDeepSeek最终回复：{final_response.content}")
    
else:
    print(f"DeepSeek：{ai_msg.content}")