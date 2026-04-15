import os
from dotenv import load_dotenv  # 引入我们刚安装的包
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 施展魔法：读取 .env 文件
# 这行代码会自动把你 .env 里的东西塞进系统的环境变量里
load_dotenv() 

# 你看，这里我们就不需要写 os.environ[...] 了！
# 因为 LangChain 里的 ChatOpenAI 非常聪明，它在初始化的时候，
# 会自动去系统的环境变量里找有没有 OPENAI_API_KEY 和 OPENAI_API_BASE 这两个名字的值。

# 2. 实例化“超级大脑”
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0.7,  # 控制回答的创造性，0.7表示平衡（0更保守，1更发散）
    max_tokens=512    # 限制回答的最大长度，防止生成过长内容
)

# 3. 制定规矩（提示词模板）
template = """
你是一个幽默的编程老师。
请用一句话给新手解释什么是 {tech_concept}？
"""
prompt = PromptTemplate.from_template(template)

# 4. 把“模板”和“大脑”串起来
chain = prompt | llm

# 5. 执行！
print("思考中...\n")
response = chain.invoke({"tech_concept": "Python里的环境变量(.env)"})

# 6. 打印回答
print("DeepSeek说：")
print(response.content)