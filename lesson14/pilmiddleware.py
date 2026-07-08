from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

# 创建带PII脱敏能力的智能体
agent = create_agent(
    model=llm,
    tools=[],  # 此处可传入实际工具
    system_prompt="你是一个贴心的客服助手。",
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
            strategy="mask"
        ),
        PIIMiddleware("email", strategy="mask",apply_to_input=True),
    ]
)

# 测试：用户输入中包含手机号和邮箱
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "我的手机号是13812345678，邮箱是test@example.com，请帮我查一下订单。"
            }
        ]
    }
)
print(result['messages'][-1].content)