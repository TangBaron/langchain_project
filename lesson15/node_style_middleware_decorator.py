from datetime import datetime
from typing import Any
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """当消息数量超过1条时直接结束对话"""
    if len(state["messages"]) >= 2:
        return {
            "last_call_at": datetime.now().isoformat(),
            "messages": [AIMessage(content="对话已达到上限，请重新开始。")],
            "jump_to": "end"
        }
    return None

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        check_message_limit
    ],
    checkpointer=InMemorySaver()
)

while True:
    content = input('用户：')
    result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "user_1"}},
    )
    print('大模型：', result['messages'][-1].content)
    if '对话已达到上限' in result['messages'][-1].content:
        print('当前的状态是:', result)
        break






