run benchmark script on Core or Xeon

```
llm-convert "./llm-models/chatglm2-6b/" --model-format pth --model-family "chatglm" --outfile "checkpoint/"
llm-convert "./llm-models/llama-2-13b-chat-hf/" --model-format pth --model-family "llama" --outfile "checkpoint/"
```
```
cd benchmark

conda activate llm

source bigdl-nano-init -c

./test_llm_spr.sh > llm_memory1.log 2>&1 
```
