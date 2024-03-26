
## Chapter 7 Finetune

As one of the advanced parameter-efficient fine-tuning (PEFT) techniques, QLoRA enables light-weight infusion of specialty knowledge into a large language model with minimal overhead. IPEX-LLM also supports finetuning LLM (large language models) using QLora with 4bit optimizations on Intel GPUs.

> **Note**
>
> Currently, IPEX-LLM supports LoRA, QLoRA, ReLoRA, QA-LoRA and DPO finetuning.

In Chapter 7, you will go through how to fine-tune a large language model to a text generation task using IPEX-LLM optimizations. IPEX-LLM has a comprehensive tool-set to help you fine-tune the model, merge the LoRA weights and inference with the fine-tuned model.

We are going to train with a popular open source model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example. For other finetuning methods, please refer to the [LLM-Finetuning](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/LLM-Finetuning) page for detailed instructions.

## 7.0 Environment Setup

Please refer to the [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for mode details. It is strongly recommended that you follow the corresponding steps below to configure your environment properly.