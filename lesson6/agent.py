import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools import all_tools

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="你注册的api-key"
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

model_with_tools = model.bind_tools(all_tools)


def run_agent_with_tools(query: str, max_iterations: int = 30) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for i in range(max_iterations):
        print("🤖 正在等待 AI 思考...")

        # Step 1: 调用模型
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # Step 2: 检查是否有工具调用
        if not response.tool_calls or len(response.tool_calls) == 0:
            print(f"\n✨ AI 最终回复:\n{response.content}\n")
            return response.content

        # Step 3: 执行工具调用
        for tool_call in response.tool_calls:
            # 在工具列表中查找对应的工具
            found_tool = None
            for t in all_tools:
                if t.name == tool_call["name"]:
                    found_tool = t
                    break

            if found_tool:
                # 执行工具
                tool_result = found_tool.invoke(tool_call["args"])
                # 将结果包装为 ToolMessage 回传
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

    return "⚠️ 已达到最大迭代次数，任务可能未完成。"


def run_agent(query: str):
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=all_tools
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content=query)
        ]
    })
    return response
