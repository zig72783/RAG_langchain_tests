#加载LLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers.generation.utils import GenerationConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

device = "cuda" # the device to load the model onto

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

prompt = "HMT是什么"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = llm.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


if isinstance(model_inputs.input_ids, torch.Tensor):
    print("model_inputs.input_ids is a valid torch.Tensor")
else:
    print("model_inputs.input_ids is not a torch.Tensor")



#构建RAG
from langchain.document_loaders import PyPDFLoader

###加载文件
loader = PyPDFLoader("https://arxiv.org/pdf/2405.06067")
pages = loader.load()


len(pages[0].page_content)


###文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50,)

docs = text_splitter.split_documents(pages[:4])

len(docs[0].page_content)



HUGGINGFACEHUB_API_TOKEN = "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"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

db = FAISS.from_documents(docs, embeddings)

query = "What is HMT?"
# retriever = db.similarity_search(query , k = 3)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# source_knowledge = "\n".join([x.page_content for x in result_simi])
# 构建prompt
#augmented_prompt = """Using the contexts below, answer the query.

#contexts:
#{source_knowledge}

#query: {query}"""


# 创建prompt
#prompt = PromptTemplate(template=augmented_prompt, input_variables=["source_knowledge" ,"query"])


# llm_chain = LLMChain(prompt=prompt, llm=llm  , llm_kwargs = {"temperature":0, "max_length":1024})*************

#llm_chain = prompt | llm | StrOutputParser()

# response = llm_chain.invoke({"source_knowledge":source_knowledge ,"query" : query })

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke({"question": "What is Task Decomposition?"})

print(response)


# llm_chain = ({"source_knowledge": source_knowledge, "query": RunnablePassthrough()} | prompt | llm  | StrOutputParser())

#response = llm_chain.invoke({"source_knowledge": source_knowledge, "query": query})

#print(response)


#augmented_prompt_2 = f"""Using the contexts below, answer the query.

#contexts:
#{source_knowledge}

#query: {query}"""

#print(augmented_prompt_2)
