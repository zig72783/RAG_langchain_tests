## 本地加载LLM
#import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers.generation.utils import GenerationConfig

#import torch
#from modelscope import snapshot_download, Model
#model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master')
#model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
#messages = []
#messages.append({"role": "user", "content": "HMT是什么”"})
#response = model(messages)
#print(response)


## 本地加载Qwen
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

prompt = "什么是HMT"
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

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

