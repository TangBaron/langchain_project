'''
# 消息类型的演示代码

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间。

    Args:
        format: 时间格式字符串，默认为 "%Y-%m-%d %H:%M:%S"

    Returns:
        当前时间的字符串表示
    """
    current_time = datetime.now()
    return current_time.strftime(format)

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[get_current_time],
    system_prompt="你是一个有用的助手，可以使用工具来帮助用户解决问题。"
)


if __name__ == "__main__":
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": "今天是几月几日呢？"}
        ]
    })

    # 1. 获取content内容
    for msg in response['messages']:
        if msg.content and msg.content != '':
            if isinstance(msg, HumanMessage):
                print('用户输入消息：', msg.content)
            elif isinstance(msg, AIMessage):
                print('大模型输出消息：', msg.content)
            elif isinstance(msg, ToolMessage):
                print('工具输出信息:', msg.content)

    print('\n\n')

    # 2. 获取工具调用信息
    for msg in response['messages']:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"Tool: {tool_call['name']}")
                print(f"Args: {tool_call['args']}")
                print(f"ID: {tool_call['id']}")
        if isinstance(msg, ToolMessage):
            print('ToolMessage的工具调用id', msg.tool_call_id)

    print('\n\n')

    # 3. 获取本次对话token用量信息
    for msg in response['messages']:
        if isinstance(msg, AIMessage):
            print('打印本次大模型的用量信息:', msg.usage_metadata)

'''

# 多模态演示代码
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

import base64
import mimetypes
from pathlib import Path

def image_to_base64(image_path: str, as_data_uri: bool = False) -> str:
    """
    将本地图片文件转换为 Base64 字符串

    :param image_path: 图片文件路径
    :param as_data_uri: 是否返回带有 data URI 前缀的字符串（例如 data:image/png;base64,...）
    :return: Base64 字符串
    """
    # 检查文件是否存在
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {image_path}")

    # 读取图片二进制数据
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 进行 Base64 编码，并转为字符串
    base64_str = base64.b64encode(image_data).decode("utf-8")

    # 如果需要 Data URI 前缀
    if as_data_uri:
        # 自动获取 MIME 类型（根据文件扩展名）
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            # 如果无法识别，默认为 application/octet-stream
            mime_type = "application/octet-stream"
        data_uri = f"data:{mime_type};base64,{base64_str}"
        return data_uri
    else:
        return base64_str

image_data = image_to_base64('test.png')

# message = HumanMessage(
#     content=[
#         {"type": "text", "text": "请描述一下这张图片的内容"},
#         {
#             "type": "image",
#             "base64": image_data,
#             "mime_type": "image/png"
#         }
#     ]
# )

message = HumanMessage(
    content_blocks=[
        { "type": "text", "text": "请描述一下这张图片的内容" },
        {
            "type": "image",
            "base64": image_data,
            "mime_type": "image/png"
        }
    ]
)

response = llm.invoke([message])
print(response)















