## pip install transformers_stream_generator einops tiktoken

from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM,AutoModel
import torch
#import intel_extension_for_pytorch as ipex


#model_name = "chatglm2-6b"
#model_name = "Baichuan2-7B-Chat" # "Qwen-7B-Chat"
model_name = "internlm-chat-7b-8k"
model_all_local_path = "./checkpoint/"
model_name_local = model_all_local_path + model_name

if model_name == "chatglm2-6b":
    tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_local,trust_remote_code=True,load_in_4bit=True)
  #  model = model.cpu()
    model.save_low_bit(model_all_local_path + "/chatglm2-6b-int4/")
    tokenizer.save_pretrained(model_all_local_path + "/chatglm2-6b-int4/")

elif model_name == "Llama-2-7b-chat-hf" or model_name == "StarCoder" or model_name == "Qwen-7B-Chat" or model_name == "Baichuan2-7B-Chat" or model_name == "internlm-chat-7b-8k":
    tokenizer = AutoTokenizer.from_pretrained(model_name_local,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_local,trust_remote_code=True,load_in_4bit=True)
   # model = model.cpu()
    model.save_low_bit(model_all_local_path + model_name + "-int4/")
    tokenizer.save_pretrained(model_all_local_path + model_name + "-int4/")