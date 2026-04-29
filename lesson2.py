from datetime import datetime
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

    # 输出运行过程的消息列表
    print(response)
