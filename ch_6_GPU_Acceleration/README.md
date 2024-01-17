# Chapter 6 GPU Acceleration

Apart from the significant acceleration capabilites on Intel CPUs, BigDL-LLM also supports optimizations and acceleration for running LLMs (large language models) on Intel GPUs.

BigDL-LLM supports optimizations of any [*HuggingFace transformers*](https://huggingface.co/docs/transformers/index) model on Intel GPUs with the help of low-bit techniques, modern hardware accelerations and latest software optimizations.

#### 6B model running on Intel Arc GPU (real-time screen capture):

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif" width='60%' /> 

</p>

#### 13B model running on Intel Arc GPU (real-time screen capture): 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif" width='60%' /> 

</p>

In Chapter 6, you will learn how to run LLMs, as well as implement stream chat functionalities, using BigDL-LLM optimizations on Intel GPUs. Popular open source models are used as examples:

+ [Llama2-7B](./6_1_GPU_Llama2-7B.md)


## 6.0 System Support
### 1. Linux: 
**Hardware**:
- Intel Arc™ A-Series Graphics
- Intel Data Center GPU Flex Series
- Intel Data Center GPU Max Series

**Operating System**:
- Ubuntu 20.04 or later (Ubuntu 22.04 is preferred)

### 2. Windows
For Windows, BigDL-LLM is verified for [HF-Transformers-AutoModels]([./HF-Transformers-AutoModels/](https://github.com/intel-analytics/BigDL/tree/1ba8d99956eace6f506037375f45b11f64dde7a6/python/llm/example/GPU/HF-Transformers-AutoModels)) and [PyTorch-Models]([./PyTorch-Models/](https://github.com/intel-analytics/BigDL/blob/1ba8d99956eace6f506037375f45b11f64dde7a6/python/llm/example/GPU/PyTorch-Models)) on Intel iGPU and dGPU.

**Hardware**:
- Intel iGPU and dGPU

**Operating System**:
- Windows 10/11, with or without WSL 


## 6.1 Environment Setup

Please refer to the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details. It is strongly recommended that you follow the corresponding steps below to configure your environment properly.