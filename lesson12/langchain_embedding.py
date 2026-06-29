from langchain_community.embeddings import DashScopeEmbeddings

embedding_model = DashScopeEmbeddings(
    model='text-embedding-v4',
    dashscope_api_key='',

)

def get_embedding_vector(text):
    return embedding_model.embed_query(text)

if __name__ == '__main__':
    print(get_embedding_vector('你好'))
    print(len(get_embedding_vector('你好')))
