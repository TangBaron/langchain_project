from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import json
import re

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="",
)


# 1. 定义输出格式（Pydantic 模型）
class WeatherReport(BaseModel):
    """天气报告结构化输出"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="当前温度，单位摄氏度")
    condition: str = Field(description="天气状况，如晴朗、多云、小雨等")
    suggestion: str = Field(description="根据天气给出的一句出行建议")


# 2. 定义工具
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息
    params:
    - city: 要查询的城市名称
    """
    return f"{city}今天25度，天气晴朗"


# 3. 创建智能体，不指定 response_format，避免触发 tool_choice 限制
agent = create_agent(
    model=llm,
    tools=[get_weather],
)

# 在 system prompt 中要求模型按 JSON 格式输出最终结果
system_prompt = f"""你是一个天气助手。请按以下步骤执行：
1. 使用可用工具查询天气信息。
2. 根据工具返回的结果，以 JSON 格式输出最终答案，不要输出任何其他解释文字。

JSON 必须严格符合以下 schema：
{json.dumps(WeatherReport.model_json_schema(), indent=2, ensure_ascii=False)}

注意：只输出纯 JSON，不要加 markdown 代码块标记（如 ```json）。"""


# 4. 调用
result = agent.invoke({
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]
})

# 5. 获取最后一条消息的文本内容并解析为结构化结果
last_content = result["messages"][-1].content

# 去除可能的 markdown 代码块
text = last_content.strip()
text = re.sub(r"^```(?:json)?\s*", "", text)
text = re.sub(r"\s*```$", "", text)

weather = WeatherReport.model_validate_json(text.strip())

print(f"城市：{weather.city}")
print(f"温度：{weather.temperature}°C")
print(f"天气：{weather.condition}")
print(f"建议：{weather.suggestion}")
