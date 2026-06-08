from agent_mcp_http import run_agent

def main():
    print("=" * 50)
    print("[MCP-HTTP] mini-cursor Python 版")
    print("基于 LangChain + MCP (HTTP/SSE) 的 AI 编程助手")
    print("=" * 50)
    print()
    print("提示: 请先启动 MCP Server: python tool_mcp_http.py")
    print()
    query = input("请输入你的需求: ").strip()

    if not query:
        print("请输入有效的需求描述")
        return

    print()
    print(f"[任务] {query}")
    print("-" * 50)
    print()

    result = run_agent(query)

    print()
    print("=" * 50)
    print("[完成] 任务结束")
    print("=" * 50)


if __name__ == "__main__":
    main()
