import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
#from langchain_core.runnables import RunnablePassthrough

from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


###加载文件
loader = PyPDFLoader("https://arxiv.org/pdf/2405.06067")
pages = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

###文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50,)

docs = text_splitter.split_documents(pages[:4])

###加载embedding模型
model_name = "sentence-transformers/sentence-t5-large"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

###加载faiss向量库
query = "HMT是什么?"
db = FAISS.from_documents(docs, embeddings)
retriever = db.similarity_search(query , k = 3)
source_knowledge = "\n".join([x.page_content for x in retriever])

###LLM部署
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

# Prompt
question = "什么是HMT"
prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

retriever = db.as_retriever()

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

response = rag_chain.invoke(question)
print(response)
