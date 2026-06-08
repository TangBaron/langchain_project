import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

model = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=''
)

SYSTEM_PROMPT = f"""
你是一个项目管理助手，使用工具完成任务。

当前工作目录: {os.getcwd()}

工具:
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令 (支持 working_directory 参数)
4. list_directory: 列出目录

重要规则 - execute_command:
- workingDirectory 参数会自动切换到指定目录
- 当使用 workingDirectory 时，绝对不要在 command 中使用 cd
- 错误示例: {{ command: "cd project && npm install", workingDirectory: "project" }}
  这是错误的！因为 workingDirectory 已经在 project 目录了，再 cd project 会找不到目录
- 正确示例: {{ command: "npm install", workingDirectory: "project" }}
  这样就对了！workingDirectory 已经切换到 project，直接执行命令即可

回复要简洁，只说做了什么。
"""

# 显式指定 HTTP/SSE 传输模式，连接到本地 MCP Server
mcp_client = MultiServerMCPClient(
    {
        "agent-tools": {
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    }
)

LOG_FILE = "mcp_server.log"


def print_server_logs():
    """读取并打印 Server 日志文件中的新内容"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                print("[Server 日志]")
                for line in lines:
                    print(f"  {line.rstrip()}")
                # 清空日志文件，避免重复打印
                open(LOG_FILE, "w", encoding="utf-8").close()
    except Exception:
        pass


async def run_agent_with_mcp(query: str, max_iterations: int = 30) -> str:
    tools = await mcp_client.get_tools()
    print(f"已加载 {len(tools)} 个工具: {[t.name for t in tools]}")

    model_with_tools = model.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for i in range(max_iterations):
        print("[Agent] 正在等待 AI 思考...")

        # Step 1: 调用模型
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # Step 2: 检查是否有工具调用
        if not response.tool_calls or len(response.tool_calls) == 0:
            print(f"\n[Agent] AI 最终回复:\n{response.content}\n")
            return response.content

        # Step 3: 执行工具调用
        for tool_call in response.tool_calls:
            found_tool = None
            for t in tools:
                if t.name == tool_call["name"]:
                    found_tool = t
                    break

            if found_tool:
                print(f"[Agent] 调用工具: {tool_call['name']}({tool_call['args']})")
                tool_result = await found_tool.ainvoke(tool_call["args"])
                print(f"[Agent] 工具返回: {str(tool_result)[:200]}...")
                # 打印 Server 日志
                print_server_logs()
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

    return "[警告] 已达到最大迭代次数，任务可能未完成。"


def run_agent(query: str):
    return asyncio.run(run_agent_with_mcp(query))


if __name__ == "__main__":
    query = "你好"
    if query:
        result = run_agent(query)
        print(result)
    else:
        print("请输入有效的需求描述")
