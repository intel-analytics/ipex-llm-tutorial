# Chapter 1 Introduction

## What is BigDL-LLM
[BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm) is a library that makes LLMs (language language models) run fast on mid-to-low-end PCs. It is released as part of the open source project [BigDL](https://github.com/intel-analytics/bigdl) with Apache 2.0 License. 


## What can you do with BigDL-LLM
You can use BigDL-LLM to run _any [Huggingface transformer](https://huggingface.co/docs/transformers/index) model_. It impllictly quantize and accelerate the heavy parts of the original pytorch model, employing modern hardware accelerations as well as latest software optimizations. 

Transformer-API based applications can be easily changed to use BigDL-LLM with minor code change. And you'll immediately observe a speedup after using BigDL-LLM. 

```python
# change import, specify precision when loading the model
from bigdl.llm.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/model/', load_in_4bit=True)
# no code change needed in model inference
...
```

BigDL-LLM supports a variety of low-precion optimizations, e.g. INT4/INT5/INT8. It allows you to run larger LLM models on PCs with limited resources. For exmaple, you will be able to run a 7B or 13B model on a 16G memory laptop with fast enough experience. 

#### 7B model running on a 12-Gen Core PC (real-time screen apture):

![image](https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-7b.gif)

#### 13B model running on a 12-Gen Core PC (real-time screen apture): 

![image](https://github.com/bigdl-project/bigdl-project.github.io/blob/master/assets/llm-13b.gif)


## Quick Installation

```bash
pip install bigdl-llm[all]
```
## What's Next

The following chapters in this tutorial will explain in more details about how to use BigDL-LLM to run LLMs, e.g. transformer API, langchain APIs, Chinese support, etc. Each chapter will provide runnable notebooks using popular open source models as example. Read along to learn more and play. 


Also, you can check out our [GitHub repo](https://github.com/intel-analytics/BigDL/tree/main/python/llm) for more information and latest news.

We have varified a lot of models using BigDL-LLM and made them runnable examples, including [Llama2](), [Vicuna](), [ChatGLM](), [ChatGLM2](), etc. You can find model examples [here](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/transformers/transformers_int4).

