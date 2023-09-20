# Chapter 1 Introduction

## What is BigDL-LLM
[BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm) is a low-bit LLM library on Intel XPU (Xeon/Core/Flex/Arc/PVC), featuring broadest model support, lowest latency and smallest memory footprint. It is released as part of the open source [BigDL](https://github.com/intel-analytics/bigdl) project under Apache 2.0 License. 


## What can you do with _BigDL-LLM_
You can use BigDL-LLM to run _any pytorch model_ (e.g. [HuggingFace transformer](https://huggingface.co/docs/transformers/index) models). It automatically optimizes and accelerates LLMs using low-bit optimizations, modern hardware accelerations and latest software optimizations. 

Using BigDL-LLM is easy. With just 1-line of code change, you can immediately observe significant speedup [^1] . 

#### Example: Optimize LLaMA model with `optimize_model`
```python
from bigdl.llm import optimize_model

from transformers import LlamaForCausalLM, LlamaTokenizer
model = LlamaForCausalLM.from_pretrained(model_path,...)

# apply bigdl-llm low-bit optimization, by default uses INT4
model = optimize_model(model)

...
```

BigDL-LLM provides a variety of low-bit optimizations (e.g., INT3/NF3/INT4/NF4/INT5/INT8), and allows you to run LLMs on low-cost PCs (CPU-only), on PCs with GPU, or on cloud. 

The demos below shows the experiences of running 7B and 13B model on a 16G memory laptop.  

#### 6B model running on an Intel 12-Gen Core PC (real-time screen capture):

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" width='60%' /> 

</p>

#### 13B model running on an Intel 12-Gen Core PC (real-time screen capture): 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif" width='60%' /> 

</p>


Installation is also super easy with just 1-line `pip install`. [Chapter 2: Enviroment Setup](../ch_2_Environment_Setup/README.md) talks about best practices for preparing your environment.

## What's Next

The following chapters in this tutorial will explain in more details about how to use BigDL-LLM to build LLM applications, e.g. APIs, Chinese support, GPU, application development guides with case studies. Each chapter will provide runnable notebooks using popular open source models. Read along to learn more and run the code on your laptop.


Also, you can check out our [GitHub repo](https://github.com/intel-analytics/BigDL/tree/main/python/llm) for more information and latest news.

We have already verified many models on BigDL-LLM and provided ready-to-run examples, such as [Llama](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [Llama2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/llama2), [Vicuna](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/vicuna), [ChatGLM](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm), [ChatGLM2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2), [Baichuan](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/baichuan), [MOSS](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/moss), [Falcon](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/falcon), [Dolly-v1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v1), [Dolly-v2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v2), StarCoder([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/starcoder)), Phoenix([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4),[link2](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4/phoenix)), RedPajama([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/redpajama)), [Whisper](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/whisper), etc. You can find model examples [here](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4).


[^1]: Performance varies by use, configuration and other factors. `bigdl-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.

