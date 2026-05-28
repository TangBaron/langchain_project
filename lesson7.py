import asyncio
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="你注册的阿里云百炼api_key"
)

mcp_client = MultiServerMCPClient(
    {
        "amap-maps": {
              "command": "cmd",
              "args": [
                "/c",
                "npx",
                "-y",
                "@amap/amap-maps-mcp-server"
              ],
              "env": {
                "AMAP_MAPS_API_KEY": "你注册的高德地图api_key"
              },
              'transport': 'stdio'
            }
    }
)

async def get_server_tools():
    tools = await mcp_client.get_tools()
    print(f"加载了{len(tools)}: {[t.name for t in tools]}")



asyncio.run(get_server_tools())

async def build_agent():
    mcp_tools = await mcp_client.get_tools()
    print(f"加载了{len(mcp_tools)}: {[t.name for t in mcp_tools]}")
    agent_with_mcp = create_agent(
        llm,
        tools=mcp_tools,
        system_prompt = "你是一个高德地图规划助手，能帮我规划形成和获得地图基本信息"
    )
    result = await agent_with_mcp.ainvoke(
        {
            "messages":{
                "role": 'user',
                "content": "请告诉我北京圆明园到北京西北旺地铁站距离"
            }
        }
    )
    for msg in result['messages']:
        msg.pretty_print()


asyncio.run(build_agent())

