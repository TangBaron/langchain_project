from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from typing import Annotated, List, Callable


def skill_list_reducer(current: List[str], new: List[str]) -> List[str]:
    """
    累积模式：合并已加载的 Skills
    保持所有已加载的技能，而不是替换
    """
    if not current:
        return new
        # 合并并去重，保持顺序
    combined = current + [s for s in new if s not in current]
    return combined


# 使用累积模式的 reducer
class SkillState(MessagesState):
    """
    Skill 状态 Schema
    """
    skills_loaded: Annotated[List[str], skill_list_reducer] = []


from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.tools import tool


# ==================== Loader 工具 ====================
# 这些工具始终可见，用于加载其他技能

@tool
def skill_data_analysis(runtime) -> Command:
    """
    加载数据分析技能。
    """
    instructions = """
        数据分析技能已成功加载！
    
        现在你可以使用以下工具：
        • calculate_statistics(numbers): 计算一组数字的统计信息
        • generate_chart(data, chart_type): 生成数据图表
    
        请继续使用这些工具完成用户的数据分析任务。
    """

    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["data_analysis"]  # 关键：直接更新状态
        }
    )


@tool
def skill_text_processing(runtime) -> Command:
    """
    加载文本处理技能。

    调用此工具后，你将获得以下文本处理相关的工具：
    - summarize_text: 生成文本摘要
    - extract_keywords: 提取关键词

    使用场景：当用户需要处理文本、生成摘要或提取关键信息时，
    请先调用此工具加载文本处理技能。
    """
    instructions = """
        文本处理技能已成功加载！
    
        现在你可以使用以下工具：
        • summarize_text(text, max_length): 生成文本摘要
        • extract_keywords(text, num_keywords): 提取关键词
        
        请继续使用这些工具完成用户的文本处理任务。
    """

    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["text_processing"]  # 关键：直接更新状态
        }
    )


# ==================== 数据分析工具 ====================
# 这些工具只有在加载了 data_analysis 技能后才可见
@tool
def calculate_statistics(numbers: List[float]) -> str:
    """
    计算一组数字的统计信息，包括平均值、最大值、最小值、标准差等。

    Args:
        numbers: 要分析的数字列表
    """
    import statistics

    if not numbers:
        return "错误: 数字列表为空"

    result = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }

    if len(numbers) > 1:
        result["stdev"] = statistics.stdev(numbers)

    return f"统计结果: {result}"


@tool
def generate_chart(data: List[float], chart_type: str = "bar") -> str:
    """
    根据数据生成图表（模拟）。

    Args:
        data: 数据列表
        chart_type: 图表类型 (bar, line, pie)
    """
    return f"已生成 {chart_type} 图表，包含 {len(data)} 个数据点"


# ==================== 文本处理工具 ====================
# 这些工具只有在加载了 text_processing 技能后才可见

@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    生成文本摘要。

    Args:
        text: 要摘要的文本
        max_length: 摘要最大长度
    """
    if len(text) <= max_length:
        return f"摘要: {text}"
    return f"摘要: {text[:max_length]}..."


@tool
def extract_keywords(text: str, num_keywords: int = 5) -> str:
    """
    从文本中提取关键词。

    Args:
        text: 要分析的文本
        num_keywords: 要提取的关键词数量
    """
    # 简单模拟：取前几个单词
    words = text.split()[:num_keywords]
    return f"关键词: {', '.join(words)}"


# 组织工具
LOADER_TOOLS = [skill_data_analysis, skill_text_processing]
DATA_ANALYSIS_TOOLS = [calculate_statistics, generate_chart]
TEXT_PROCESSING_TOOLS = [summarize_text, extract_keywords]
ALL_TOOLS = LOADER_TOOLS + DATA_ANALYSIS_TOOLS + TEXT_PROCESSING_TOOLS

from langchain.tools import BaseTool

# 技能到工具的映射
SKILL_TOOL_MAPPING = {
    "data_analysis": DATA_ANALYSIS_TOOLS,
    "text_processing": TEXT_PROCESSING_TOOLS,
}


def get_tools_for_skills(skills_loaded: List[str]) -> List[BaseTool]:
    """
    根据已加载的技能列表，返回应该暴露给模型的工具

    核心逻辑：
    1. Loader 工具始终包含
    2. 根据 skills_loaded 添加对应的技能工具

    Args:
        skills_loaded: 已加载的技能名称列表

    Returns:
        过滤后的工具列表
    """
    # 始终包含 Loader 工具
    tools = list(LOADER_TOOLS)

    # 根据已加载的技能添加对应工具
    for skill_name in skills_loaded:
        if skill_name in SKILL_TOOL_MAPPING:
            tools.extend(SKILL_TOOL_MAPPING[skill_name])

    return tools


class SkillMiddleware(AgentMiddleware):
    """
    Skill 中间件 - 实现动态工具过滤

    这是 Claude Skills 的核心组件！

    工作原理：
    1. 在每次模型调用前拦截请求
    2. 从 request.state 中读取 skills_loaded 列表
    3. 根据 skills_loaded 过滤工具列表
    4. 使用 request.override() 替换工具列表
    5. 传递给下一个 handler

    这样，模型在每次调用时只会看到相关的工具！
    """

    def __init__(self, verbose: bool = True):
        """
        初始化 SkillMiddleware

        Args:
            verbose: 是否打印详细日志（用于调试和演示）
        """
        super().__init__()
        self.verbose = verbose
        self.call_count = 0

    def _get_skills_from_state(self, request: ModelRequest) -> List[str]:
        """
        从请求状态中提取 skills_loaded

        注意：AgentState 是 TypedDict，本质上是 dict
        所以我们使用字典方式访问
        """
        skills_loaded = []

        if hasattr(request, 'state') and request.state is not None:
            # TypedDict 本质是 dict，使用 .get() 方法
            if isinstance(request.state, dict):
                skills_loaded = request.state.get("skills_loaded", [])
            else:
                # 兼容其他类型
                skills_loaded = getattr(request.state, "skills_loaded", [])

        return skills_loaded

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        【核心方法】拦截模型调用，动态过滤工具

        这是整个 Claude Skills 系统最关键的方法！
        """
        self.call_count += 1

        # Step 1: 从状态中获取已加载的 Skills
        skills_loaded = self._get_skills_from_state(request)

        # Step 2: 获取过滤后的工具
        filtered_tools = get_tools_for_skills(skills_loaded)

        # Step 3: 打印日志
        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"[SkillMiddleware] 第 {self.call_count} 次模型调用")
            print(f"{'─' * 60}")
            print(f"skills_loaded: {skills_loaded}")
            print(f"过滤后工具 ({len(filtered_tools)}个): {[t.name for t in filtered_tools]}")

            # 对比原始工具数量
            if hasattr(request, 'tools') and request.tools:
                original_count = len(request.tools)
                print(f"工具数量变化: {original_count} → {len(filtered_tools)}")

        # Step 4: 【关键】使用 request.override() 替换工具列表
        # 这会创建一个新的 ModelRequest，其中 tools 被替换为过滤后的列表
        filtered_request = request.override(tools=filtered_tools)

        if self.verbose:
            print(f"已将过滤后的工具传递给模型")
            print(f"{'─' * 60}\n")

        # Step 5: 调用下一个 handler（实际的模型调用）
        return handler(filtered_request)


skill_middleware = SkillMiddleware(verbose=True)

llm = ChatOpenAI(
    model="qwen3.8-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

SYSTEM_PROMPT = """
你是一个智能助手，可以使用各种技能来帮助用户完成任务。

## 工作方式

1. 你有两类工具：
   - **Skill Loader**（技能加载器）：用于加载特定技能，名称以 skill_ 开头
   - **功能工具**：执行具体任务的工具

2. 当用户请求某个功能时：
   - 首先检查是否有对应的功能工具
   - 如果没有，调用相应的 Skill Loader 加载技能
   - 加载后，使用新获得的工具完成任务

3. 可用的 Skill Loaders：
   - skill_data_analysis：加载数据分析相关工具
   - skill_text_processing：加载文本处理相关工具

请根据用户的需求，灵活使用工具完成任务。
"""

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    middleware=[skill_middleware],
    state_schema=SkillState,
    system_prompt=SYSTEM_PROMPT
)

'''
# 测试场景一：当用户请求数据分析时，观察工具的动态加载过程。
test_input = {
    "messages": [HumanMessage(content="我有一组销售数据 [150, 200, 180, 220, 190]，请帮我计算统计信息")],
    "skills_loaded": []  # 初始状态：没有加载任何技能
}

# 调用 Agent
result = agent.invoke(test_input)

print("-" * 60)
print("\n最终状态:")
print(f"   skills_loaded: {result.get('skills_loaded', [])}")

print("\nAI 响应:")
for msg in result.get("messages", []):
    if msg.__class__.__name__ == "AIMessage" and msg.content:
        print(msg.content)
'''

# 测试场景二：同时需要数据分析和文本处理的场景
test_input = {
    "messages": [{
        "role": "user",
        "content": """请帮我完成以下任务：
1. 计算这组数据的统计信息: [85, 92, 78, 95, 88]
2. 从这段文本中提取关键词: "人工智能正在改变各行各业的工作方式"
"""
    }],
    "skills_loaded": []
}

# 调用 Agent
result = agent.invoke(test_input)

print("-"*60)
print("\n最终状态:")
print(f"   skills_loaded: {result.get('skills_loaded', [])}")

print("\nAI 响应:")
for msg in result.get("messages", []):
    if msg.__class__.__name__ == "AIMessage" and msg.content:
        print(msg.content)