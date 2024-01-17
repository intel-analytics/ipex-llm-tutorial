
## Chapter 7 Finetune

As one of the advanced parameter-efficient fine-tuning (PEFT) techniques, QLoRA enables light-weight infusion of specialty knowledge into a large language model with minimal overhead. BigDL-LLM also supports finetuning LLM (large language models) using QLora with 4bit optimizations on Intel GPUs.

> **Note**
>
> Currently, BigDL-LLM only supports QLoRA finetuning on any [Hugging Face `transformers` models](https://huggingface.co/docs/transformers/index).

In Chapter 7, you will go through how to fine-tune a large language model to a text generation task using BigDL-LLM optimizations. BigDL-LLM has a comprehensive tool-set to help you fine-tune the model, merge the LoRA weights and inference with the fine-tuned model.

We are going to train with a popular open source model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example.


## 7.0 System Support

**Hardware**:
- Intel Arc™ A-Series Graphics
- Intel Data Center GPU Flex Series
- Intel Data Center GPU Max Series

**Operating System**:
- Ubuntu 20.04 or later (Ubuntu 22.04 is preferred)


## 7.1 Environment Setup

Please refer to the [GPU installation guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details. It is strongly recommended that you follow the corresponding steps below to configure your environment properly.