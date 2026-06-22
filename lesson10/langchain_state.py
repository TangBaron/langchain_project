from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState


class CustomAgentState(AgentState):
    user_name: str
    description: str


llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="你是一个有用的助手，可以帮助用户解决各种问题。同时当用户表明自己的名字时，应该将该名字写入到状态中",
    state_schema=CustomAgentState,

)

result = agent.invoke(
    {
        "messages": {
            "role": 'user',
            "content": "你好，我叫什么名字?"
        },
        'user_name': '大模型真好玩',
        'description': '一名科技博主'
    }
)

print(result['user_name'])
print(result['description'])
print(result['messages'])
