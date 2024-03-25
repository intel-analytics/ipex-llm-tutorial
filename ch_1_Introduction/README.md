# Chapter 1 Introduction

## What is IPEX-LLM
[IPEX-LLM](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm) is a low-bit LLM library on Intel XPU (Xeon/Core/Flex/Arc/PVC), featuring broadest model support, lowest latency and smallest memory footprint. It is released under Apache 2.0 License. 


## What can you do with _IPEX-LLM_
You can use IPEX-LLM to run _any pytorch model_ (e.g. [HuggingFace transformers](https://huggingface.co/docs/transformers/index) models). It automatically optimizes and accelerates LLMs using low-bit optimizations, modern hardware accelerations and latest software optimizations. 

Using IPEX-LLM is easy. With just 1-line of code change, you can immediately observe significant speedup [^1] . 

#### Example: Optimize LLaMA model with `optimize_model`
```python
from ipex_llm import optimize_model

from transformers import LlamaForCausalLM, LlamaTokenizer
model = LlamaForCausalLM.from_pretrained(model_path,...)

# apply ipex-llm low-bit optimization, by default uses INT4
model = optimize_model(model)

...
```

IPEX-LLM provides a variety of low-bit optimizations (e.g., INT3/NF3/INT4/NF4/INT5/INT8), and allows you to run LLMs on low-cost PCs (CPU-only), on PCs with GPU, or on cloud. 

The demos below shows the experiences of running 7B and 13B model on a 16G memory laptop.  

#### 6B model running on an Intel 12-Gen Core PC (real-time screen capture):

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" width='60%' /> 

</p>

#### 13B model running on an Intel 12-Gen Core PC (real-time screen capture): 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif" width='60%' /> 

</p>



## What's Next

The following chapters in this tutorial will explain in more details about how to use IPEX-LLM to build LLM applications, e.g. best practices for setting up your environment, APIs, Chinese support, GPU, application development guides with case studies, etc. Most chapters provide runnable notebooks using popular open source models. Read along to learn more and run the code on your laptop.


Also, you can check out our [GitHub repo](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm) for more information and latest news.

We have already verified many models on IPEX-LLM and provided ready-to-run examples, such as [Llama2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2), [Vicuna](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna), [ChatGLM](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm), [ChatGLM2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2), [Baichuan](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan), [MOSS](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/moss), [Falcon](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon), [Dolly-v1](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1), [Dolly-v2](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2), [StarCoder](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder), [Mistral](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral), [RedPajama](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama), [Whisper](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model/whisper), etc. You can find more model examples [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Model).


[^1]: Performance varies by use, configuration and other factors. `ipex-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.

