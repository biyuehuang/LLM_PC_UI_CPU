## step 1
##模型文件夹名称是 checkpoint，共19GB，
##     包括三个Native INT4模型bigdl_llm_llama2_13b_q4_0.bin,bigdl_llm_starcoder_q4_0.bin, ggml-chatglm2-6b-q4_0.bin

## 修改本脚本第285行 main函数里的模型存放路径，例如  model_all_local_path = "C:/Users/username/checkpoint/"

## 代码文件 LLM_demo_v1.0.py.py和theme3.json


# step 2
#conda create -n llm python=3.9
#conda activate llm
#pip install --pre --upgrade bigdl-llm[all]
#pip install gradio mdtex2html torch
#python LLM_demo_v1.0.py

## 如果在任务管理器里CPU大核没有用起来，因为应用变成后端运行了。
## 你可以试一下用管理员打开Anaconda Powershell Prompt窗口

## 调整机器的性能模式，一个是Windows自带的电源》最佳性能
## 另一个是每个OEM厂商定义的性能模式，可以从厂商提供的电脑管家之类的应用里面找找。如性能调节，内存优化

## UI参数说明
## 1.温度（Temperature）（数值越高，输出的随机性增加）
## 2.Top P（数值越高，词语选择的多样性增加）
## 3.输出最大长度（Max Length）（输出文本的最大tokens）

from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer,TextStreamer
import gradio as gr
import mdtex2html
import argparse
import time
from bigdl.llm.transformers import AutoModelForCausalLM
import torch
import sys
import gc
import os
import psutil
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM
from bigdl.llm.transformers import BigdlNativeForCausalLM
from transformers import TextIteratorStreamer
#from bigdl.llm import optimize_model

DICT_FUNCTIONS = {
 #   "聊天助手":     "问：{prompt}\n\n答：",
    "聊天助手":     "{prompt}\n",
 #   "生成大纲":     "帮我生成一份{prompt}的大纲\n\n",
 #   "情感分析":     "对以下内容做情感分析：{prompt}\n\n",
 #   "信息提取":     "对以下内容做精简的信息提取：{prompt}\n\n", 
 #   "中文翻译":     "将以下内容翻译成英文：{prompt}\n\n",
 #   "美食指南":     "请提供{prompt}的食谱和烹饪方法\n\n",
 #   "故事创作":     "讲一个关于{prompt}的故事\n\n",
 #   "旅游规划":     "请提供{prompt}的旅游规划\n\n"
}


##显示当前 python 程序占用的内存大小
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('******************* {} memory used: {} MB'.format(hint, memory))

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpu", type=str, default="cpu")
    args = parser.parse_args()
    return args

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def stream_chat(model, tokenizer, prompt, input, max_new_tokens, history=[]):
    input_ids = tokenizer([prompt], return_tensors='pt')

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # skip prompt in the generated tokens
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens
    )
    
    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    from threading import Thread   
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    history = []

    output_str = ""
    for stream_output in streamer:
        output_str += stream_output
       # print(output_str)
        yield output_str#, history


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
   # yield text
    return text

# LLama2 Starcoder-15.5b load 
def load(model_path, model_family, n_threads,n_ctx):
    llm = BigdlNativeForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        model_family=model_family,
        n_threads=n_threads,
        n_ctx=n_ctx)
    return llm

#def predict(input, function, chatbot, max_length, top_p, temperature, history, past_key_values,model_select):
def predict(input, function, chatbot, max_length, top_p, temperature, history, model_select):
    global model_name,model_all_local_path,model,tokenizer
    input = parse_text(input)
    if model_select != model_name:
        print("********** Switch model from ",model_name,"to",model_select)
        model_name = model_select      
        del model
        gc.collect()
        show_memory_info('after del old model')

        stm = time.time()
        try:
            if model_name == "chatglm2-6b":
                print("******* loading chatglm2-6b")
                ## https://github.com/intel-analytics/BigDL/blob/main/python/llm/src/bigdl/llm/ggml/model/chatglm/chatglm.py
                model = ChatGLM(model_all_local_path + "\\ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096) #use_mmap=False, n_threads=20, n_ctx=512

            elif model_name == "llama2-13b":
                print("******* loading llama2-13b")
                model = load(model_path=model_all_local_path + "\\bigdl_llm_llama2_13b_q4_0.bin",
                        model_family='llama',
                        n_threads=20,n_ctx=4096)

            elif model_name == "Qwen-7b-chat":
                print("******* loading Qwen-7B")         
                model = AutoModelForCausalLM.load_low_bit(model_all_local_path + "\\Qwen-7B-Chat-int4", trust_remote_code=True, optimize_model=False)
              #  model = BenchmarkWrapper(model)
                tokenizer = AutoTokenizer.from_pretrained(model_all_local_path + "\\Qwen-7B-Chat-int4", trust_remote_code=True)

               # from bigdl.llm import optimize_model
               # model = optimize_model(model)
            elif model_name == "Baichuan2-7b-chat":
                print("******* loading Baichuan2-7b-chat")         
                model = AutoModelForCausalLM.load_low_bit(model_all_local_path + "\\Baichuan2-7B-Chat-int4", trust_remote_code=True, optimize_model=False)
              #  model = BenchmarkWrapper(model)
                tokenizer = AutoTokenizer.from_pretrained(model_all_local_path + "\\Baichuan2-7B-Chat-int4", trust_remote_code=True)
            elif model_name == "internlm-chat-7b-8k":
                print("******* loading internlm-chat-7b-8k")         
                model = AutoModelForCausalLM.load_low_bit(model_all_local_path + "\\internlm-chat-7b-8k-int4", trust_remote_code=True, optimize_model=False)
              #  model = BenchmarkWrapper(model)
                tokenizer = AutoTokenizer.from_pretrained(model_all_local_path + "\\internlm-chat-7b-8k-int4", trust_remote_code=True)
        except:
            print("******************** Can't find local model ************************")
            sys.exit(1)  
        print("********** model load time (s)= ", time.time() - stm)  
        show_memory_info('after load new model')  
       
    ## refer: https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2/README.md
    chatbot.append((input, ""))    
    response = ""
    timeFirst = 0
    timeFirstRecord = False
    
    if model_name == "chatglm2-6b" or model_name == "llama2-13b":
        template = DICT_FUNCTIONS[function]
        prompt = template.format(prompt=input)
        timeStart = time.time()
        for chunk in model(prompt, temperature=temperature,top_p=top_p, stream=True,max_tokens=max_length):
            response += chunk['choices'][0]['text']
            chatbot[-1] = (input, parse_text(response))
            if timeFirstRecord == False:
                timeFirst = time.time() - timeStart
                timeFirstRecord = True
            yield chatbot, "", "", ""  

    elif model_name == "Qwen-7b-chat" or model_name == "Baichuan2-7b-chat" or model_name == "internlm-chat-7b-8k":
        template4 = DICT_FUNCTIONS[function]
        prompt = template4.format(prompt=input)
        
        with torch.inference_mode():
            input_ids = tokenizer([prompt], return_tensors='pt')

            streamer = TextIteratorStreamer(tokenizer,
                                            skip_prompt=True, # skip prompt in the generated tokens
                                            skip_special_tokens=True)
            timeStart = time.time()
            generate_kwargs = dict(
                input_ids,
                streamer=streamer,
                max_new_tokens=max_length
            )
            
            # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
            from threading import Thread   
            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()
            history = []

            response = ""
            for stream_output in streamer:
                response += stream_output
                chatbot[-1] = (input, parse_text(response))
               	if timeFirstRecord == False:
               	
               	    timeFirst = time.time() - timeStart
                    timeFirstRecord = True
                yield chatbot, "",  "", ""  
                

            # for response in stream_chat(model, tokenizer, prompt, input, max_new_tokens=max_length):              
            #   #  print("****", response)
            #     chatbot[-1] = (input, parse_text(response))
            #    	if timeFirstRecord == False:
               	
            #    	    timeFirst = time.time() - timeStart
            #         timeFirstRecord = True
            #     yield chatbot, "",  "", ""  
           

    timeCost = time.time() - timeStart

    if model_name == "Qwen-7b-chat" or model_name == "Baichuan2-7b-chat" or model_name == "internlm-chat-7b-8k":
        token_count_input = len(tokenizer.tokenize(prompt))
        token_count_output = len(tokenizer.tokenize(response))
    else:
        token_count_input = len(model.tokenize(prompt))  
        token_count_output = len(model.tokenize(response))   
   
    ms_first_token = timeFirst * 1000
    ms_after_token = (timeCost - timeFirst) / (token_count_output - 1) * 1000
    print("input: ", prompt)
    print("output: ", parse_text(response))
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("time cost(s): ", timeCost)
    print("First token latency(ms): ", ms_first_token)
    print("After token latency(ms/token)", ms_after_token)
    print("-"*40)
    yield chatbot, history, str(round(ms_first_token, 2)) + " ms", str(round(ms_after_token, 2)) + " ms/token"

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], "", ""

css="""
body{display:flex;} 
.radio-group .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
}
footer {visibility: hidden}
"""

if __name__ == '__main__':
    args = getArgs()
    xpu = args.xpu
    model_name = "None"
    #model_name = "chatglm2-6b"
    model_all_local_path = "./checkpoint"
    model = None
    tokenizer = None
    
    """Override Chatbot.postprocess"""
    gr.Chatbot.postprocess = postprocess

    # Read function titles
    listFunction = list(DICT_FUNCTIONS.keys())

    # Main UI Framework display:flex;flex-wrap:wrap;
    with gr.Blocks(theme=gr.themes.Base.load("theme3.json"), css=css) as demo: ## 可以在huging face下载模板
        gr.HTML("""<h1 align="center">英特尔大语言模型应用</h1>""")
        with gr.Row():
            with gr.Column(scale=2.5):
                user_function = gr.Radio(listFunction, elem_classes="radio-group", label="功能", value=listFunction[0], min_width=120, scale=1, interactive=True)
                with gr.Column(scale=1, visible=True): # 配置是否显示控制面板                       
                    model_select = gr.Dropdown(["chatglm2-6b","llama2-13b","Qwen-7b-chat","Baichuan2-7b-chat","internlm-chat-7b-8k"],value="chatglm2-6b",label="选择模型", interactive=True)
                    device_inf = gr.Dropdown(["CPU"],value="CPU",label="推理设备", interactive=True)
                    max_length = gr.Slider(0, 2048, value=512, step=1.0, label="输出最大长度", interactive=True)                       
                    temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                    with gr.Column():
                        f_latency = gr.Textbox(label="First Latency", visible=True)
                        a_latency = gr.Textbox(label="After Latency", visible=True)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(scale=1)
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(show_label=False, placeholder="请在此输入文字...", lines=5, container=False, scale=5,interactive=True)
                        with gr.Row():
                            submitBtn = gr.Button("提交", variant="primary",interactive=True)
                            emptyBtn = gr.Button("清除",interactive=True)
                        gr.Examples( [ "你好","我最近经常失眠，晚上睡不着怎么办？","我计划国庆节出去玩一周，给我推荐一下江浙沪周边3个目的地","帮我生成一份大语言模型对未来的影响的论文的大纲","将以下内容翻译成英文：条条大路通罗马","请提供红烧狮子头的食谱和烹饪方法","推荐上海三个旅游景点","以第一人称视角介绍太阳的起源和变化","如何提升个人魅力",
                            "What is AI?","Add 1 and 3, which gives us","Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people and have fun."],user_input,chatbot)
        
     
        history = gr.State([])
        # Action for submit/empty button
        submitBtn.click(predict, [user_input, user_function, chatbot, max_length, top_p, temperature, history, model_select],
                        [chatbot, history, f_latency, a_latency], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history, f_latency, a_latency], show_progress=True)
    # Launch the web app
    demo.queue().launch(share=False, inbrowser=True)