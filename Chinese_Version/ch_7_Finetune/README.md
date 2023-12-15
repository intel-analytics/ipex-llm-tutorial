# 第七章 微调

作为最新的参数微调方法，QLoRA 能够在仅微调少量参数情况下高效地将专业知识注入到预训练后的大语言模型中。BigDL-LLM 同样支持使用QLora 在英特尔 GPU 上进行 4 bit 优化来微调 LLM（大语言模型）。

> **注意**
>
> 目前仅支持Hugging Face Transformers模型运行QLoRA微调，且BigDL-LLM支持在英特尔GPU上优化任何 [*HuggingFace Transformers*](https://huggingface.co/docs/transformers/index)模型。
> 目前，BigDL-LLM 仅支持对[Hugging Face `transformers` 模型](https://huggingface.co/docs/transformers/index)进行 QLoRA 微调。

在第7章中，您将了解如何使用 BigDL-LLM 优化将大型语言模型微调适配文本生成任务。BigDL-LLM 可帮助您微调模型、进行 LoRA 权重与基本权重的合并以及应用合并后的模型进行推理。

我们将以当下流行的开源模型 [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) 为例进行训练。

## 7.0 环境设置

您可以按照[第6章](./ch_6_GPU_Acceleration/README.md) 中的详细说明在英特尔 GPU 上配置环境。以下仅列举正确配置环境的**必要**步骤。

### 7.0.1 系统需求
> ⚠️硬件
   - 英特尔 Arc™ A 系列显卡
   - 英特尔数据中心 GPU Flex 系列
   - 英特尔数据中心 GPU Max 系列

> ⚠️操作系统
   - Linux系统，Ubuntu 22.04优先

### 7.0.2 安装驱动程序和工具包

在英特尔 GPU 上使用 BigDL-LLM 之前，有几个安装工具的步骤：

- 首先，您需要安装英特尔 GPU 驱动程序。请参阅我们的驱动程序安装以了解更多关于通用 GPU 功能的事项。

- 您还需要下载并安装英特尔® oneAPI Base Toolkit。OneMKL 和 DPC++ 编译器是必选项，其他为可选项。

### 7.0.3 Python环境配置

假设您已经安装了[Conda](https://docs.conda.io/projects/conda/en/stable/)作为您的python环境管理工具，下面的命令可以帮助您创建并激活您的 Python 环境：

````bash
# 建议使用 Python 3.9 来运行 BigDL-LLM
conda create -n llm-finetune python=3.9
conda activate llm-finetune 
````

### 7.0.4 设置OneAPI环境变量

您需要在 Intel GPU 上为 BigDL-LLM 设置 OneAPI 环境变量。

```bash
# 配置 OneAPI 环境变量
source /opt/intel/oneapi/setvars.sh
```

### 7.0.5（可选项） 在英特尔 GPU 上运行模型推理的配置

如果您想使用英特尔 GPU 对微调后的模型进行推理，建议设置以下变量以达到最佳性能：

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```