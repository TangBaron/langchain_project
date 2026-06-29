from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import IndexConfig
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

embedding_model = DashScopeEmbeddings(
    model='text-embedding-v4',
    dashscope_api_key='',
)

def get_embedding_vector(texts):
    return [embedding_model.embed_query(text) for text in texts]

store = InMemoryStore(index=IndexConfig(embed=get_embedding_vector, dims=1024))


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[RuntimeContext]) -> str:
    """保存用户信息到长期记忆"""
    user_id = runtime.context.user_id
    namespace = ("users", user_id)  # 命名空间：按用户维度组织
    # 将用户信息存入 store
    runtime.store.put(namespace, user_id, dict(user_info))
    print(f"已成功保存用户 {user_info['name']} 的信息！")
    return f"已成功保存用户 {user_info['name']} 的信息！"


@tool
def search_user_memory(query: str, runtime: ToolRuntime[RuntimeContext]) -> str:
    """在用户的长期记忆中搜索相关信息"""
    print('正在查询...')
    user_id = runtime.context.user_id
    namespace = ("users", user_id)  # 限定搜索范围

    # 向量相似度搜索
    results = runtime.store.search(
        namespace,
        query=query,  # 向量查询
    )

    if results:
        return f"找到相关记忆：{results[0].value}"
    else:
        return "未找到相关记忆"

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[save_user_info, search_user_memory],
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
# 输出：已保存用户 张三 的信息

thread_config = {"configurable": {"thread_id": "2"}}

# ----- 第三次对话（换了新的 thread_id，但长期记忆依然有效）-----
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "我的偏好有哪些"}]},
    config=thread_config,
    context=RuntimeContext(user_id="user_123"),
)
print('第三次对话: \n', result3['messages'][-1].content)

