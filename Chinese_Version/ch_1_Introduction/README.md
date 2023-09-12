# 第一章 简介

## 什么是 BigDL-LLM
[BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm) 是一个库，可使 LLM（大型语言模型）在低成本 PC 上快速[^1]运行（无需独立显卡）。它作为开源项目 [BigDL](https://github.com/intel-analytics/bigdl) 的一部分发布，采用 Apache 2.0 许可。


## 您能用 BigDL-LLM 做些什么
您可以使用 BigDL-LLM 运行 _任何 [HuggingFace transformer](https://huggingface.co/docs/transformers/index) 模型_。它使用低精度技术、现代硬件加速和最新软件优化自动优化和加速 LLM。

只需修改一行代码，基于 HuggingFace transformer 的应用程序就能在 BigDL-LLM 上运行，而且您会立即观察到显著的速度提升[^1]。

```python
# 修改 import，在加载模型的时候指定精度
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)
# 模型推理部分的代码无需修改
...
```

BigDL-LLM 提供各种低精度优化（如 INT4/INT5/INT8），允许您在资源有限的 PC 上运行 LLM。例如，您可以在 16G 内存的笔记本电脑上运行 7B 或 13B 模型，而且延迟非常低[^1]。

#### 在英特尔 12 代酷睿电脑上运行 6B 模型（实时屏幕画面）:

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" width='60%' /> 

</p>

#### 在英特尔 12 代酷睿电脑上运行 13B 模型（实时屏幕画面）: 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif" width='60%' /> 

</p>


## 快速安装

```bash
pip install bigdl-llm[all]
```
在安装 BigDL-LLM 之前，建议先安装 Python 3.9 和 conda。阅读 [第二章：环境配置](../ch_2_Environment_Setup/README.md) 了解准备环境的最佳做法。

## 接下来

本教程以下各章将详细介绍如何使用 BigDL-LLM 构建 LLM 应用程序，例如 transformers API、langchain API、多语言支持等。每一章都将使用流行的开源模型提供可运行的笔记本。您可以继续阅读以了解更多信息，同时也可以在您的笔记本电脑上运行提供的代码。

此外，您还可以访问我们的 [GitHub repo](https://github.com/intel-analytics/BigDL/tree/main/python/llm) 获取更多信息和最新消息。

我们已经在 BigDL-LLM 上验证了很多的模型并且提供了可立即运行的示例，例如 [Llama](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [Llama2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/llama2), [Vicuna](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/vicuna), [ChatGLM](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm), [ChatGLM2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2), [Baichuan](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/baichuan), [MOSS](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/moss), [Falcon](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/falcon), [Dolly-v1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v1), [Dolly-v2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v2), StarCoder([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/starcoder)), Phoenix([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4),[link2](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4/phoenix)), RedPajama([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/redpajama)), [Whisper](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/whisper) 等。你可以在[这里](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4)找到模型的示例。


[^1]: 性能因使用、配置和其他因素而异。对于非英特尔产品，`bigdl-llm`的优化程度可能不同。了解更多信息，请访问 www.Intel.com/PerformanceIndex。

