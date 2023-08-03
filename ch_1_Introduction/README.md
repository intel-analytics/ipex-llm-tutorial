# Chapter 1 Introduction

## What is BigDL-LLM
[BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm) is a library that makes LLMs (language language models) run fast on low-cost PCs (without the need of discrete GPU). It is released as part of the open source [BigDL](https://github.com/intel-analytics/bigdl) project under Apache 2.0 License. 


## What can you do with BigDL-LLM
You can use BigDL-LLM to run _any [HuggingFace transformer](https://huggingface.co/docs/transformers/index) model_. It automatically optimizes and accelerates LLMs using low-precision techniques, modern hardware accelerations and latest software optimizations. 

HuggingFace transformers-based applications can run on BigDL-LLM with one-line code change, and you'll immediately observe significant speedup. 

```python
# change import, specify precision when loading the model
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)
# no code change needed in model inference
...
```

BigDL-LLM provides a variety of low-precision optimizations (e.g., INT4/INT5/INT8), and allows you to run LLMs on PCs with limited resources. For example, you will be able to run a 7B or 13B model on a 16G memory laptop with very low latency.  

#### 7B model running on an Intel 12-Gen Core PC (real-time screen capture):

![image](https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-7b.gif)

#### 13B model running on an Intel 12-Gen Core PC (real-time screen capture): 

![image](https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-13b.gif)


## Quick Installation

```bash
pip install bigdl-llm[all]
```
Python 3.9 and conda are recommended for installing BigDL-LLM. Read [Chapter 2: Quick Start]() to learn more about the best practices. 

## What's Next

The following chapters in this tutorial will explain in more details about how to use BigDL-LLM to build LLM applications, e.g. transformers API, langchain APIs, multi-language support, etc. Each chapter will provide runnable notebooks using popular open source models. Read along to learn more and run the code on your laptop.


Also, you can check out our [GitHub repo](https://github.com/intel-analytics/BigDL/tree/main/python/llm) for more information and latest news.

We have already verified many models on BigDL-LLM and provided ready-to-run examples, such as [Llama](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [Llama2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/llama2), [Vicuna](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/vicuna), [ChatGLM](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm), [ChatGLM2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2), [Baichuan](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/baichuan), [MOSS](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/moss), [Falcon](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/falcon), [Dolly-v1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v1), [Dolly-v2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v2), StarCoder([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/starcoder)), Phoenix([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4),[link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/redpajama)), RedPajama([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/redpajama)), [Whisper](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/whisper), etc. You can find model examples [here](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4).

> **NOTE**  
> BigDL-LLM is optimized on Intel laptops, and performance on other platforms may not be optimal.

