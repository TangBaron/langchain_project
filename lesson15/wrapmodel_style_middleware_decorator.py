from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain.agents.middleware import ModelRequest, ModelResponse, ExtendedModelResponse
from langchain_core.messages import SystemMessage, AIMessage
from typing import Callable
from datetime import datetime

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

@wrap_model_call
def inject_time_and_annotate(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    """
    1. 修改请求：在消息列表最前面插入一条包含当前时间的系统消息
    2. 调用原始模型
    3. 修改响应：在模型回复的内容后追加一个注释
    """
    # ---------- 1. 修改请求 ----------
    # 构造一条包含当前时间的系统消息（使用 SystemMessage 对象）
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_message = SystemMessage(content=f"当前时间是 {now}。请在回答时充分考虑时间因素。")
    
    # 使用 override() 方法创建新的请求，将时间消息插入到消息列表开头
    modified_request = request.override(
        messages=[time_message] + request.messages
    )

    # ---------- 2. 调用原始模型 ----------
    response = handler(modified_request)

    # ---------- 3. 修改响应 ----------
    # ModelResponse.result 是消息列表，找到最后一条 AIMessage 并修改其内容
    modified_results = []
    for msg in response.result:
        if isinstance(msg, AIMessage) and msg.content:
            # 在内容后追加注释
            new_content = msg.content + "\n\n（处理完成）"
            modified_results.append(AIMessage(content=new_content))
        else:
            modified_results.append(msg)

    # 创建新的 ModelResponse
    modified_response = ModelResponse(
        result=modified_results,
        structured_response=response.structured_response # 一般只有指定格式化才会需要这个参数
    )

    # 返回 ExtendedModelResponse
    return ExtendedModelResponse(
        model_response=modified_response
    )

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        inject_time_and_annotate
    ],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": '你好'}]},
    config={"configurable": {"thread_id": "user_1"}},
)

print(result['messages'])