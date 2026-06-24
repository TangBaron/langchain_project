from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langgraph_sdk.schema import Context
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime


class CustomAgentState(AgentState):
    user_name: str
    description: str


llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

# 使用 Pydantic 定义 Context 的结构
class MyContext(BaseModel):
    user_id: str
    user_name: str
    department: str = "unknown"  # 可设置默认值

@tool
def get_user_info(runtime: ToolRuntime[MyContext]) -> str:
    """获取当前用户信息"""
    # runtime.context 是 Context 类型的 Pydantic 模型
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name
    return f"当前用户：{user_name}（ID: {user_id}）"

agent = create_agent(
    model=llm,
    tools=[get_user_info],
    system_prompt="你是一个有用的助手。",
    context_schema=MyContext,    # 指定 Context 结构
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请根据我的身份回答问题"}]},
    context=MyContext(
        user_id="user_123",
        user_name="张三",
        department="技术部"
    )
)
print(result['messages'][-1])