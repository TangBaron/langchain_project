import sys
from agent import run_agent_with_tools, run_agent

def main():
    # 交互模式
    print("=" * 50)
    print("🚀 mini-cursor Python 版")
    print("基于 LangChain 1.0 的 AI 编程助手")
    print("=" * 50)
    print()
    query = input("💬 请输入你的需求: ").strip()

    if not query:
        print("❌ 请输入有效的需求描述")
        return

    print()
    print(f"📋 任务: {query}")
    print("-" * 50)
    print()

    # 运行 Agent
    result = run_agent_with_tools(query)

    print()
    print("=" * 50)
    print("✅ 任务完成")
    print("=" * 50)

def main_agent():
    # 交互模式
    print("=" * 50)
    print("🚀 mini-cursor Python 版")
    print("基于 LangChain 1.0 的 AI 编程助手")
    print("=" * 50)
    print()
    query = input("💬 请输入你的需求: ").strip()

    if not query:
        print("❌ 请输入有效的需求描述")
        return

    print()
    print(f"📋 任务: {query}")
    print("-" * 50)
    print()

    # 运行 Agent
    result = run_agent(query)

    print()
    print("=" * 50)
    print("✅ 任务完成")
    print("=" * 50)


if __name__ == "__main__":
    #main()
    main_agent()