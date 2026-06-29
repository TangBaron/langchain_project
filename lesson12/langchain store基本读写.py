from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

# ----- 定义 Context（用户 ID 作为只读配置） -----
class RuntimeContext(BaseModel):
    user_id: str

# ----- 定义用户信息的结构（用于工具的参数校验） -----
class UserInfo(TypedDict):
    name: str
    language: str
    preferences: list[str]

store = InMemoryStore()


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[RuntimeContext]) -> str:
    """保存用户信息到长期记忆"""
    user_id = runtime.context.user_id
    namespace = ("users",)  # 命名空间：按用户维度组织
    # 将用户信息存入 store
    runtime.store.put(namespace, user_id, dict(user_info))
    return f"已成功保存用户 {user_info['name']} 的信息！"

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[save_user_info],
    system_prompt="你是一个贴心的助手，请记住用户告诉你的个人信息。",
    store=store,
    context_schema=RuntimeContext,
)

# 第一次对话：用户告诉智能体自己的信息
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫张三，我平时用中文交流，喜欢简短直白的回答"}]},
    context=RuntimeContext(user_id="user_123")
)

print(result['messages'][-1].content)

# 直接访问 store 验证数据
saved = store.get(("users",), "user_123")
print(saved.value)  # 输出：{'name': '张三', 'language': '中文', 'preferences': ['喜欢简短直白的回答']}

