from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Any
from datetime import datetime, timedelta


# 1. 定义扩展状态
class RateLimitState(AgentState):
    user_id: str = "anonymous"
    call_history: List[str] = []  # 存储调用时间戳


# 2. 定义限流中间件（类模式）
class RateLimitMiddleware(AgentMiddleware):
    """基于滑动窗口的用户级限流中间件"""

    def __init__(self, window_seconds: int = 60, max_calls: int = 10):
        """
        Args:
            window_seconds: 时间窗口长度（秒）
            max_calls: 窗口内最大允许调用次数
        """
        self.window_seconds = window_seconds
        self.max_calls = max_calls
        super().__init__()

    @hook_config(can_jump_to=["end"])
    def before_model(
            self,
            state: RateLimitState,
            runtime: Runtime
    ) -> dict[str, Any] | None:
        """每次模型调用前检查限流"""
        user_id = state.get("user_id", "anonymous")
        history = state.get("call_history", [])

        # 清理过期的调用记录（超出时间窗口的）
        now = datetime.now().isoformat()
        cutoff = (datetime.now() - timedelta(seconds=self.window_seconds)).isoformat()
        recent_calls = [t for t in history if t > cutoff]

        # 检查是否超限
        if len(recent_calls) >= self.max_calls:
            return {
                "messages": [
                    AIMessage(
                        content=f"⚠️ 调用频率超限！用户 {user_id} 在 {self.window_seconds}秒内 "
                        f"已调用 {self.max_calls} 次，请稍后再试。"
                    )
                ],
                "jump_to": "end"
            }

        # 未超限：记录本次调用时间
        recent_calls.append(now)
        return {"call_history": recent_calls}

    # 异步版本
    async def abefore_model(
            self,
            state: RateLimitState,
            runtime: Runtime
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


llm = ChatOpenAI(
    model="qwen3.7-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[RateLimitMiddleware(window_seconds=1000, max_calls=2)],
    state_schema=RateLimitState,
    checkpointer=InMemorySaver()
)

# 模拟快速连续调用
thread_config = {"configurable": {"thread_id": "test_user"}}
for i in range(5):
    result = agent.invoke(
        {"messages": [HumanMessage(f"请回复数字 {i+1}")]},
        config=thread_config
    )
    content = result['messages'][-1].content
    print(f"第{i+1}次调用结果：{content[:50]}...")
    print("-" * 40)
    # 如果触发限流，退出循环
    if "调用频率超限" in content:
        print("检测到限流，结束循环。")
        break
