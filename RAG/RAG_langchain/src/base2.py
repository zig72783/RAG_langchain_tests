import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
    revision="v2.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", revision="v2.0")
messages = []
# messages.append({"role": "user", "content": "解释一下“温故而知新”"})
messages.append({"role": "user", "content": "HMT是什么"})
response = model.chat(tokenizer, messages)
print(response)
# "温故而知新"是一句中国古代的成语，出自《论语·为政》篇。这句话的意思是：通过回顾过去，我们可以发现新的知识和理解。换句话说，学习历史和经验可以让我们更好地理解现在和未来。

# 这句话鼓励我们在学习和生活中不断地回顾和反思过去的经验，从而获得新的启示和成长。通过重温旧的知识和经历，我们可以发现新的观点和理解，从而更好地应对不断变化的世界和挑战。
