from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.postgres import PostgresStore
from typing_extensions import TypedDict

# ----- 定义 Context（用户 ID 作为只读配置） -----
class RuntimeContext(BaseModel):
    user_id: str

# ----- 定义用户信息的结构（用于工具的参数校验） -----
class UserInfo(TypedDict):
    name: str
    language: str
    preferences: list[str]


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[RuntimeContext]) -> str:
    """保存用户信息到长期记忆"""
    user_id = runtime.context.user_id
    namespace = ("users", user_id)  # 命名空间：按用户维度组织
    # 将用户信息存入 store
    runtime.store.put(namespace, user_id, dict(user_info))
    print(f"已成功保存用户 {user_info['name']} 的信息！")
    return f"已成功保存用户 {user_info['name']} 的信息！"

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    agent = create_agent(
        model=llm,
        tools=[save_user_info],
        system_prompt="你是一个贴心的助手，请记住用户告诉你的个人信息。",
        store=store,
        context_schema=RuntimeContext,
        checkpointer=InMemorySaver()
    )

    thread_config = {"configurable": {"thread_id": "1"}}

    # ----- 第一次对话（新用户）-----
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好！"}]},
        config=thread_config,
        context=RuntimeContext(user_id="user_123")
    )
    print('第一次对话: \n', result1['messages'][-1].content)
    # 输出：我还不认识你，请告诉我你的信息

    # ----- 第二次对话（用户告诉姓名）-----
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫大模型真好玩，我喜欢用中文交流"}]},
        config=thread_config,
        context=RuntimeContext(user_id="user_123")
    )
    print('第二次对话: \n', result2['messages'][-1].content)
    # 输出：已保存用户 大模型真好玩 的信息

