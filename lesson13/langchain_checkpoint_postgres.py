from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver


llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

# 配置会话线程
thread_config = {"configurable": {"thread_id": "1"}}
DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="你是一个有用的助手，可以帮助用户解决各种问题。",
        checkpointer=checkpointer
    )
    result1 = agent.invoke(
        {
            "messages": [
                {
                    "role": 'user',
                    "content": "你好，我叫大模型真好玩?"
                }
            ]
        },
        config=thread_config
    )

    print('输出第一次调用结果:', result1['messages'][-1].content)

    result2 = agent.invoke(
        {
            "messages": [
                {
                    "role": 'user',
                    "content": "你好，我叫什么名字?"
                }
            ]
        },
        config=thread_config
    )

    print('输出第二次调用结果:', result2['messages'][-1].content)