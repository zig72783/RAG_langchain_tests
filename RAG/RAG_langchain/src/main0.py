from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")

pages = loader.load_and_split()

pages[0]

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

docs = text_splitter.split_documents(pages)

len(docs)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


embed_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model , collection_name="openai_embed")

query = "How large is the baichuan2 vocabulary?"
result = vectorstore.similarity_search(query ,k = 2)

result

def augment_prompt(query: str):
  # 获取top3的文本片段
  results = vectorstore.similarity_search(query, k=3)
  source_knowledge = "\n".join([x.page_content for x in results])
  # 构建prompt
  augmented_prompt = f"""Using the contexts below, answer the query.

  contexts:
  {source_knowledge}

  query: {query}"""
  return augmented_prompt

  print(augment_prompt(query))

# 创建prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)

messages.append(prompt)

res = chat(messages)

print(res.content)