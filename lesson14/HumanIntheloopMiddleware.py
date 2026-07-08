from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

@tool
def get_weather(loc:str)->str:
    """
    根据地点参数可以返回该地点的天气情况
    """
    return f"{loc} 天气是晴！气温23°"

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

SYSTEM_PROMPT = "你是一个天气助手，具备调用get_weather天气函数获取指定地点天气的能力，只调用一次工具，根据工具返回的结果直接回复用户"

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "get_weather": {  # 工具名
            "allowed_decisions": ["approve", "edit", "reject"],
        }
    },
    description_prefix="🚦 工具调用正在等待人工批准",
)

agent = create_agent(
    system_prompt=SYSTEM_PROMPT,
    model=llm,
    tools=[get_weather],
    middleware=[
        middleware
    ],
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎么样"}]},
    config={"configurable": {"thread_id": "user_1"}},
)

print(result)

'''
# 同意
if "__interrupt__" in result:
    print("🟠 中断已触发，人工审核中...")

    # ✅ 打印 description 信息
    for interrupt in result["__interrupt__"]:
        for req in interrupt.value["action_requests"]:
            print(f"\n📝 中断说明：{req.get('description', '无描述信息')}")

    decisions = input("请输入您的决定（approve/edit/reject）：").strip().lower()
    # 使用 APPROVE 批准调用
    result_approve = agent.invoke(
        Command(
            resume={"decisions": [{"type": decisions}]}
        ),
        config={"configurable": {"thread_id": "user_1"}}
    )

    # 打印回复内容
    print("🟢 批准后模型回复：")
    print(result_approve["messages"][-1].content)
'''

'''
# 拒绝
if "__interrupt__" in result:
    print("🟠 中断已触发，人工审核中...")

    # 🗣 显示触发中断的用户提问
    user_msg = next((m.content for m in result["messages"] if m.type == "human"), "无")
    print(f"\n🧑 用户问题：{user_msg}")

    # 🔍 打印中断详情
    for interrupt in result["__interrupt__"]:
        for i, req in enumerate(interrupt.value["action_requests"]):
            print(f"\n🔧 工具：{req['name']}")
            print(f"📦 参数：{req['args']}")
            print(f"📝 描述：{req.get('description', '无')}")
            print(f"✅ 可选操作：{interrupt.value['review_configs'][i]['allowed_decisions']}")

    # 👤 人工决定
    decisions = input("请输入您的决定（approve/edit/reject）：").strip().lower()
    if decisions not in ["approve", "edit", "reject"]:
        print("⚠️ 无效的决定，默认选择 reject。")
        decisions = "reject"

    if decisions == "reject":
        reject_message = input("请提供拒绝的理由：")
    else:
        reject_message = None

    # 🚦 发送人工决策结果
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": decisions,
                        "message": reject_message if decisions == "reject" else None
                    }
                ]
            }
        ),
        config={"configurable": {"thread_id": "user_1"}}
    )

    # 📤 显示模型后续回复
    print("\n🟢 模型后续回复：")
    print(result["messages"][-1].content)
'''

# 编辑
if "__interrupt__" in result:
    print("🟠 中断已触发，人工审核中...")

    # 🗣 显示触发中断的用户提问
    user_msg = next((m.content for m in result["messages"] if m.type == "human"), "无")
    print(f"\n🧑 用户问题：{user_msg}")

    # 🔍 打印中断详情
    for interrupt in result["__interrupt__"]:
        for i, req in enumerate(interrupt.value["action_requests"]):
            print(f"\n🔧 工具：{req['name']}")
            print(f"📦 参数：{req['args']}")
            print(f"📝 描述：{req.get('description', '无')}")
            print(f"✅ 可选操作：{interrupt.value['review_configs'][i]['allowed_decisions']}")

    # 👤 人工决定
    decisions = input("请输入您的决定（approve/edit/reject）：").strip().lower()
    if decisions not in ["approve", "edit", "reject"]:
        print("⚠️ 无效的决定，默认选择 reject。")
        decisions = "reject"

    # 根据决定类型收集额外信息
    reject_message = None
    loc = None
    if decisions == "reject":
        reject_message = input("请提供拒绝的理由：")
    elif decisions == "edit":
        loc = input("请输入正确的地点：")

    # 🚦 构建人工决策结果
    resume_data = {"decisions": [{"type": decisions}]}
    if decisions == "reject" and reject_message:
        resume_data["decisions"][0]["message"] = reject_message
    elif decisions == "edit" and loc:
        resume_data["decisions"][0]["edited_action"] = {
            "name": "get_weather",          # 工具名，通常保持不变
            "args": {"loc": loc}  # 修改后的参数
        }

    result = agent.invoke(
        Command(resume=resume_data),
        config={"configurable": {"thread_id": "user_1"}}
    )

    # 循环处理可能的多次中断（模型可能再次调用工具，因为笔者问的北京的天气，虽然中途改为上海，但是模型在回复后发现和历史记录对不上，可能会再次请求北京）
    while "__interrupt__" in result:
        print("\n🟠 再次触发中断，需要继续审批...")
        for interrupt in result["__interrupt__"]:
            for i, req in enumerate(interrupt.value["action_requests"]):
                print(f"\n🔧 工具：{req['name']}")
                print(f"📦 参数：{req['args']}")

        decisions = input("请输入您的决定（approve/edit/reject）：").strip().lower()
        if decisions not in ["approve", "edit", "reject"]:
            print("⚠️ 无效的决定，默认选择 reject。")
            decisions = "reject"

        resume_data = {"decisions": [{"type": decisions}]}
        if decisions == "reject":
            reject_message = input("请提供拒绝的理由：")
            resume_data["decisions"][0]["message"] = reject_message
        elif decisions == "edit":
            loc = input("请输入正确的地点：")
            resume_data["decisions"][0]["edited_action"] = {
                "name": "get_weather",
                "args": {"loc": loc}
            }

        result = agent.invoke(
            Command(resume=resume_data),
            config={"configurable": {"thread_id": "user_1"}}
        )

    # 📤 显示模型最终回复
    print("\n🟢 模型最终回复：")
    for msg in reversed(result["messages"]):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            print(msg.content)
            break
    else:
        print("(模型没有生成文本回复)")



