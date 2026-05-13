from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="qwen3.5-27b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=""
)

'''
# invoke 一次性输出
response = llm.invoke('你好')

print(response)
'''

'''
# stream 流式输出
for chunk in llm.stream("你好"):
    print(chunk.content, end="", flush=True)
'''

responses = llm.batch([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
])
print(responses)