import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 加载配置
load_dotenv()

# 2. 实例化“超级大脑”
# 注意：并不是所有模型都支持工具调用，但 deepseek-chat 是完美支持的
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0 # 这个值代表创造力，0比较死板，1比较天马行空
)

# 3. 打造一把“工具”（也就是一个普通的 Python 函数加上 @tool 装饰器）
# ⚠️ 注意这里面的多行注释（Docstring）极其重要！
# 大模型就是通过阅读这段注释，来决定要不要用这个工具，以及怎么填参数的。
@tool
def check_schedule(date: str) -> str:
    """
    当用户询问关于日程、安排、计划时，调用此工具。
    参数 date 是日期字符串，比如 "今天", "明天", "2026-05-01"。
    """
    print(f"\n[后台日志] 🛠️ 大模型正在悄悄调用工具，查询日期：{date}...\n")
    # 这里本来应该去查真实的数据库，我们写个假的体验一下流程
    if "今天" in date:
        return "上午10点开会，下午2点优化Python代码"
    elif "明天" in date:
        return "休息日，全天自由活动"
    else:
        return "这一天暂时没有安排"

# 4. 把工具挂载到大模型身上
# 2. bind_tools 到底做了什么？bind_tools 的英文直译就是“绑定工具”。它的核心作用，就是给这个军师递送一份“工具说明书”。
# 就像给工人发了一个工具箱，他现在拥有了查日程的能力
tools = [check_schedule]
llm_with_tools = llm.bind_tools(tools)

# 5. 测试一下！问一个它本来绝对不知道的问题
print("你：我今天有什么安排？")
response = llm_with_tools.invoke("我今天有什么安排？")

# 6. 见证奇迹的时刻
# 我们打印出大模型返回的结果，看看它到底打算怎么做
print("大模型的原始返回结果：")
print(response.tool_calls)