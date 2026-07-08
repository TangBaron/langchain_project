from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=llm,
            trigger=("messages", 5),
            keep=("messages", 3),
        ),
    ],
)


status = {
    "messages":[
        HumanMessage(content='deepseek公司最近有什么最新的资讯?', additional_kwargs={}, response_metadata={}, id='3804b3f2-827c-411a-be33-f3cc84eda168'),
        AIMessage(content='deepseek公司最近疯狂扩招', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 164, 'total_tokens': 185, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_50906f2aac', 'id': 'chatcmpl-Cid0iHZVVPLVE2vyntv2DHGuZKD4Q', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--0f8cfeb6-ec02-46cc-b24f-abdb5d29e407-0', tool_calls=[{'name': 'internet_search', 'args': {'query': 'deepseek 最新资讯', 'topic': 'news'}, 'id': 'call_CDVy2avyMw2QExcEs1f0xdNc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 164, 'output_tokens': 21, 'total_tokens': 185, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
        HumanMessage(content='deepseek公司是谁成立的?', additional_kwargs={}, response_metadata={},
                     id='3804b3f2-827c-411a-be33-f3cc84eda169'),
        AIMessage(content='deepseek公司是梁文锋成立的', additional_kwargs={'refusal': None}, response_metadata={
            'token_usage': {'completion_tokens': 21, 'prompt_tokens': 164, 'total_tokens': 185,
                            'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0,
                                                          'reasoning_tokens': 0, 'rejected_prediction_tokens': 0},
                            'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
            'model_provider': 'openai', 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_50906f2aac',
            'id': 'chatcmpl-Cid0iHZVVPLVE2vyntv2DHGuZKD4Q', 'service_tier': 'default', 'finish_reason': 'tool_calls',
            'logprobs': None}, id='lc_run--0f8cfeb6-ec02-46cc-b24f-abdb5d29e407-9', tool_calls=[
            {'name': 'internet_search', 'args': {'query': 'deepseek 最新资讯', 'topic': 'news'},
             'id': 'call_CDVy2avyMw2QExcEs1f0xdNc', 'type': 'tool_call'}],
                  usage_metadata={'input_tokens': 164, 'output_tokens': 21, 'total_tokens': 185,
                                  'input_token_details': {'audio': 0, 'cache_read': 0},
                                  'output_token_details': {'audio': 0, 'reasoning': 0}}),

    ]
}

status["messages"].append(
    HumanMessage("deepseek的新模型有哪些特点与突破?")
)

result = agent.invoke(status)
for msg in result['messages']:
    print(msg)
