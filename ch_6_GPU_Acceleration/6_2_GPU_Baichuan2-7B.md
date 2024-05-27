# 6.2 Run Baichuan 2 (7B) on Intel GPUs

You can use IPEX-LLM to load any Hugging Face *transformers* model for acceleration on Intel GPUs. With IPEX-LLM, PyTorch models (in FP16/BF16/FP32) hosted on Hugging Face can be loaded and optimized automatically on Intel GPUs with low-bit quantization (supported precisions include  INT4/NF4/INT5/FP8/INT8).

In this tutorial, you will learn how to run LLMs on Intel GPUs with IPEX-LLM optimizations, and based on that build a stream chatbot. A popular open-source LLM [baichuan-inc/Baichuan2-7B-Chat](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat) is used as an example.

## 6.2.1 Load Model in Low Precision

One common use case is to load a Hugging Face *transformers* model in low precision, i.e. conduct **implicit** quantization while loading.

For Baichuan 2 (7B), you could simply import `ipex_llm.transformers.AutoModelForCausalLM` instead of `transformers.AutoModelForCausalLM`, and specify `load_in_4bit=True` or `load_in_low_bit` parameter accordingly in the `from_pretrained` function.

For Intel GPUs, **once you have the model in low precision, set it to `to('xpu')`.**

**For INT4 Optimizations (with `load_in_4bit=True`):**

```python
from ipex_llm.transformers import AutoModelForCausalLM

# When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
# This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
model_in_4bit = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="baichuan-inc/Baichuan2-7B-Chat",
                                                     load_in_4bit=True,
                                                     trust_remote_code=True,
                                                     use_cache=True,
                                                     model_hub='modelscope')
model_in_4bit_gpu = model_in_4bit.to('xpu')
```

> **Note**
> IPEX-LLM has supported `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSpeechSeq2Seq` and `AutoModelForSeq2SeqLM`.
>
> If you have already downloaded the Baichuan 2 (7B) model, you could specify `pretrained_model_name_or_path` to the model path.


> **Note**
> * Currently, `load_in_low_bit` supports options `'sym_int4'`, `'asym_int4'`, `'sym_int5'`, `'asym_int5'` or `'sym_int8'`, in which 'sym' and 'asym' differentiate between symmetric and asymmetric quantization. Option `'nf4'` is also supported, referring to 4-bit NormalFloat. Floating point precisions `'fp4'`, `'fp8'`, `'fp16'` and mixed precisions including `'mixed_fp4'` and `'mixed_fp8'` are also supported.
>
> * `load_in_4bit=True` is equivalent to `load_in_low_bit='sym_int4'`.

## 6.2.2 Load Tokenizer 

A tokenizer is also needed for LLM inference. You can use [Huggingface transformers](https://huggingface.co/docs/transformers/index) API to load the tokenizer directly. It can be used seamlessly with models loaded by IPEX-LLM. For Baichuan 2, the corresponding tokenizer class is `AutoTokenizer`.

```python
from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="baichuan-inc/Baichuan2-7B-Chat",
                                          trust_remote_code=True)
```

> **Note**
> If you have already downloaded the Baichuan 2 (7B) model, you could specify `pretrained_model_name_or_path` to the model path.

## 6.2.3 Run Model

You can then do model inference with IPEX-LLM optimizations on Intel GPUs almostly the same way as using official `transformers` API. **The only difference is to set `to('xpu')` for token ids**. A Q&A dialog template is created for the model to complete.

```python
import torch

with torch.inference_mode():
    prompt = 'Q: What is CPU?\nA:'
    
    # tokenize the input prompt from string to token ids;
    # with .to('xpu') specifically for inference on Intel GPUs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

    # predict the next tokens (maximum 32) based on the input token ids
    output = model_in_4bit_gpu.generate(input_ids,
                            max_new_tokens=32)

    # decode the predicted token ids to output string
    output = output.cpu()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print('-'*20, 'Output', '-'*20)
    print(output_str)
```

> **Note**
> The initial generation of optimized LLMs on Intel GPUs could be slow. Therefore, it's advisable to perform a **warm-up** run before the actual generation.
>
> For the next section of stream chat, we could treat this time of generation in section 6.1.3 as a warm-up.

## 6.2.4 Stream Chat

Now, let's build a stream chatbot that runs on Intel GPUs, allowing LLMs to engage in interactive conversations. Chatbot interaction is no magic - it still relies on the prediction and generation of next tokens by LLMs. We will use Baichuan 2's built-in `chat` function to build a stream chatbot here. 

```python
chat_history = []

print('-'*20, 'Stream Chat', '-'*20, end="\n")
while True:
    prompt = input("Input: ")
    if prompt.strip() == "stop": # let's stop the conversation when user input "stop"
        print("Stream Chat with Baichuan 2 (7B) stopped.")
        break
    chat_history.append({"role": "user", "content": prompt})
    position = 0
    for response in model_in_4bit.chat(tokenizer, chat_history, stream=True):
        print(response[position:], end='', flush=True)
        position = len(response)
    print()
    chat_history.append({"role": "assistant", "content": response})
```

> **Note**
> To successfully observe the text streaming behavior in standard output, we need to set the environment variable `PYTHONUNBUFFERED=1` to ensure that the standard output streams are directly sent to the terminal without being buffered first.
