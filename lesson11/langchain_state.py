from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    resuming: int # 剩余的保修次数

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)


@tool
def use_resume(runtime: ToolRuntime):
    '''
    维修工具函数，当用户提出要保修它的产品时，调用该函数
    '''
    resume = runtime.state['resuming']
    print(f'当前剩余{resume}次保修')
    if resume > 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"已使用一次保修次数，剩余保修次数为{resume-1}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                'resuming': resume-1
            }
        )
    else:
        return '你的保修次数已经用完！'

agent = create_agent(
    model=llm,
    tools=[use_resume],
    system_prompt="你是一个维修客服，可以根据客户的需求调用工具进行维修",
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver()
)

thread_config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [{
            "role": 'user',
            "content": "你好，我想要维修我的产品"
        }],
        'resuming': 1
    },
    config=thread_config
)

print('第一次调用维修')
print(result['messages'][-1].content)

result = agent.invoke(
    {
        "messages": [
            {
                "role": 'user',
                "content": "你好, 我还想维修我的产品"
            }
        ]
    },
    config=thread_config
)

print('第二次调用维修')
print(result['messages'][-1].content)




