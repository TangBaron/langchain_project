from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware import ToolCallRequest
from langchain.tools import tool
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable, Union
from datetime import datetime

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

@wrap_tool_call
def normalize_user_id_and_timestamp(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
) -> Union[ToolMessage, Command]:
    """
    1. 修改请求：如果工具名为 "get_user_info"，则对参数中的 user_id 补零到 5 位
    2. 调用原始工具
    3. 修改响应：在工具返回的消息内容后追加当前时间戳
    """
    # ---------- 1. 修改请求 ----------
    # 仅对特定工具生效
    if request.name == "get_user_info":
        # 复制原参数，避免修改原对象（如果必要）
        modified_args = dict(request.args)  # 复制一份
        if "user_id" in modified_args:
            # 将 user_id 转换为字符串并补零到 5 位
            user_id_str = str(modified_args["user_id"]).zfill(5)
            modified_args["user_id"] = user_id_str
        # 构造新的 ToolCallRequest，用修改后的参数替换原参数
        modified_request = ToolCallRequest(
            name=request.name,
            args=modified_args,
            id=request.id,
            state=request.state,  # 保留状态引用
        )
    else:
        # 其他工具不修改
        modified_request = request

    # ---------- 2. 调用原始工具 ----------
    # 注意：handler 接收 ToolCallRequest，并返回 ToolMessage
    tool_message = handler(modified_request)

    # ---------- 3. 修改响应 ----------
    # 在工具返回的内容后追加时间戳
    original_content = tool_message.content
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_content = f"{original_content}\n[处理时间: {now}]"

    # 创建新的 ToolMessage，复制原消息的其他属性（如 tool_call_id）
    modified_tool_message = ToolMessage(
        content=new_content,
        tool_call_id=tool_message.tool_call_id,
        # 如果有其他字段，也一并复制
        additional_kwargs=tool_message.additional_kwargs,
    )

    # ---------- 返回 ----------
    # 可以选择仅返回修改后的 ToolMessage
    return modified_tool_message

    # 如果需要同时更新全局状态，可以返回 Command
    # return Command(update={"last_tool_usage": now})

@tool
def get_user_info(user_id: str) -> str:
    """根据用户ID返回用户信息（模拟）"""
    return f"用户 {user_id} 的信息：姓名-张三，年龄-28，所在地-北京"

agent = create_agent(
    model=llm,
    tools=[get_user_info],
    middleware=[normalize_user_id_and_timestamp],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "查询用户ID为123的用户信息"}]
})

# 打印最终回复（可能是工具返回的，也可能是模型综合后的）
for msg in result["messages"]:
    if isinstance(msg, ToolMessage):
        print(msg.content)