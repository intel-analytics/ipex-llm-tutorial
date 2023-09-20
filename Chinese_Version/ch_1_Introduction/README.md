# 第一章 概览

## 什么是 BigDL-LLM
[BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm) 是一个为Intel XPU(Xeon/Core/Flex/Arc/PVC)打造的低精度轻量级大语言模型库，在Intel平台上具有广泛的模型支持、最低的延迟和最小的内存占用。BigDL-LLM是开源项目 [BigDL](https://github.com/intel-analytics/bigdl) 的一部分，采用 Apache 2.0 许可证发布。


## 能用 BigDL-LLM 做什么
您可以使用 BigDL-LLM 运行任何 PyTorch 模型（例如  [HuggingFace transformer](https://huggingface.co/docs/transformers/index) 模型）。在运行过程中，BigDL-LLM利用了低精度技术、现代硬件加速技术，和一系列软件优化技术来自动加速LLM。

使用 BigDL-LLM 非常简单。只需更改一行代码，您就可以立即观察到显著的加速效果[^1]。

### 案例：使用一行`optimize_model`来优化加速LLaMA模型
```python
# 按常规流程加载LLaMA模型
from bigdl.llm import optimize_model

from transformers import LlamaForCausalLM, LlamaTokenizer
model = LlamaForCausalLM.from_pretrained(model_path,...)

# 应用BigDL-LLM 的低精度优化。默认使用 INT4
model = optimize_model(model)

# 后续模型推理部分的代码无需修改
...
```

BigDL-LLM 提供多种低精度优化（例如，INT3/NF3/INT4/NF4/INT5/INT8），并允许您使用多种平台运行LLM，包括低端笔记本（仅使用CPU）、装载Intel独立显卡的高端电脑，服务器，或者云平台。

以下演示展示了在一台16GB内存的笔记本电脑上仅使用CPU运行7B和13B模型的体验。

#### 在英特尔 12 代酷睿电脑上运行 6B 模型（实时屏幕画面）:

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-6b.gif" width='60%' /> 

</p>

#### 在英特尔 12 代酷睿电脑上运行 13B 模型（实时屏幕画面）: 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-2-13b-chat.gif" width='60%' /> 

</p>



## 接下来做什么

本教程以下各章将详细介绍如何使用 BigDL-LLM 构建 LLM 应用程序，例如 transformers API、langchain API、多语言支持等。每一章都将使用流行的开源模型提供可运行的笔记本。您可以继续阅读以了解更多信息，同时也可以在您的笔记本电脑上运行提供的代码。

此外，您还可以访问我们的 [GitHub repo](https://github.com/intel-analytics/BigDL/tree/main/python/llm) 获取更多信息和最新消息。

我们已经在 BigDL-LLM 上验证了很多的模型并且提供了可立即运行的示例，例如 [Llama](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [Llama2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/llama2), [Vicuna](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/vicuna), [ChatGLM](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm), [ChatGLM2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/chatglm2), [Baichuan](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/baichuan), [MOSS](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/moss), [Falcon](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/falcon), [Dolly-v1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v1), [Dolly-v2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/dolly_v2), StarCoder([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/starcoder)), Phoenix([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4),[link2](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4/phoenix)), RedPajama([link1](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/native_int4), [link2](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/redpajama)), [Whisper](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/transformers/transformers_int4/whisper) 等。你可以在[这里](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4)找到模型的示例。


[^1]: 性能因使用、配置和其他因素而异。对于非英特尔产品，`bigdl-llm`的优化程度可能不同。了解更多信息，请访问 www.Intel.com/PerformanceIndex。

