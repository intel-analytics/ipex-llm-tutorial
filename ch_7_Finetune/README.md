## Chapter 7 Finetune

As one of the advanced parameter-efficient fine-tuning(PEFT) techniques, QLoRA enables light-weight infusion of specialty knowledge into a large language model with minimal overhead. BigDL-LLM also supports finetuning LLMs(large language models) using QLora with 4bit optimizations on Intel GPUs.

> **Note**
>
> Currently, only Hugging Face Transformers models are supported running QLoRA finetuning, and BigDL-LLM supports optimizations of any [*HuggingFace transformers*](https://huggingface.co/docs/transformers/index) model on Intel GPUs .


In Chapter 7, you will go through how to fine-tune a large language model to a text generation task using BigDL-LLM optimizations. BigDL-LLM has a comprehensive tool-set to help you fine-tune the model, merge the LoRA weights and inference with the fine-tuned model.

We are going to train with a popular open source model [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as an example.

## 7.0 Environment Setup 

You could follow the detailed instructions in [Chapter 6](../ch_6_GPU_Acceleration/README.md) to set up your environment on Intel GPUs. Here are some **necessary** steps to configure your environment properly.

### 7.0.1 System Recommendation

> ⚠️Hardware
  - Intel Arc™ A-Series Graphics
  - Intel Data Center GPU Flex Series

> ⚠️Operating System
  - Linux system, Ubuntu 22.04 is preferred


### 7.0.2 Driver and Toolkit Installation

Before benifiting from BigDL-LLM on Intel GPUs, there’re several steps for tools installation:

- First you need to install Intel GPU driver. Please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.

- You also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.


### 7.0.3 Python Environment Setup

Supoosed that you have already installed [Conda](https://docs.conda.io/projects/conda/en/stable/) (which is recommended) as your python environment management tool, the following commands can help you create and activate your python environment: 

```bash
# Python 3.9 is recommended for running BigDL-LLM
conda create -n <your/python/environment> python=3.9 
conda activate <your/python/environment> 
```

### 7.0.4 Best Known Configuration on Linux

For optimal performance on Intel GPUs, it is recommended to set several environment variables:

```bash
# configure OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```



