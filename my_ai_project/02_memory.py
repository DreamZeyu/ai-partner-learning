"""
SystemMessage（系统消息）：给大模型设定的“人设”或背景（比如：你是个翻译官）。

HumanMessage（人类消息）：你对它说的话（你的提问）。

AIMessage（AI消息）：它回复你的话。
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. 加载环境变量 (读取你的 .env 文件)
load_dotenv()

# 2. 实例化“超级大脑”
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7
)

# 3. 准备一个小本本，用来记录聊天历史 (这就是最原始的“记忆”)
chat_history = []

# 先给它定个人设，加入历史记录
chat_history.append(SystemMessage(content="你是一个贴心的AI助手，说话非常简短。"))

print("=== 开启多轮对话 (输入 '退出' 结束) ===")

# 4. 写一个循环，让我们可以在控制台一直跟它聊天
while True:
    # 接收你的输入
    user_input = input("你：")
    
    if user_input == "退出":
        print("聊天结束，拜拜！")
        break
        
    # 将你这次说的话，打包成 HumanMessage 塞进小本本
    chat_history.append(HumanMessage(content=user_input))
    print('历史记录',chat_history)
    # 将整个小本本（包含人设、之前的聊天、你刚说的话）一起发给大模型
    response = llm.invoke(chat_history)
    
    print(f"DeepSeek：{response.content}")
    
    # ⚠️ 最重要的一步：把 AI 的回复也打包成 AIMessage 塞进小本本，这样下次它就能记住了！
    chat_history.append(AIMessage(content=response.content))