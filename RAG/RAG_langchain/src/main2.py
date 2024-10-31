# 使用API token 调用LLM
##from getpass import getpass

##HUGGINGFACEHUB_API_TOKEN = getpass("请输入您的 Hugging Face API 令牌: ")

import os

##os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# 将 API 令牌硬编码设置为环境变量
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 从环境变量中读取 API 令牌
ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

### 创建prompt 模板
question = "Where is the capital of China? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question" ])

repo_id = "baichuan-inc/Baichuan-7B"  # 具体可以参考 https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads 

llm = HuggingFaceHub(
    repo_id=repo_id, 
)
llm_chain = LLMChain(prompt=prompt, llm=llm  , llm_kwargs = {"temperature":0, "max_length":512})

print(llm_chain.run(question))



# 构建RAG检索
from langchain.document_loaders import PyPDFLoader

###加载文件
loader = PyPDFLoader("https://arxiv.org/pdf/2405.06067")
pages = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

###文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 50,)

docs = text_splitter.split_documents(pages[:4])

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

HUGGINGFACEHUB_API_TOKEN = "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"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

db = FAISS.from_documents(docs, embeddings)

query = "What is HMT?"
result_simi = db.similarity_search(query , k = 3)

source_knowledge = "\n".join([x.page_content for x in result_simi])

augmented_prompt = """Using the contexts below, answer the query.

contexts:
{source_knowledge}

query: {query}"""

prompt = PromptTemplate(template=augmented_prompt, input_variables=["source_knowledge" ,"query"])

llm_chain = LLMChain(prompt=prompt, llm=llm  , llm_kwargs = {"temperature":0, "max_length":1024})

print(llm_chain.run( {"source_knowledge":source_knowledge ,"query" : query }))

augmented_prompt_2 = f"""Using the contexts below, answer the query.

contexts:
{source_knowledge}

query: {query}"""

print(augmented_prompt_2)
