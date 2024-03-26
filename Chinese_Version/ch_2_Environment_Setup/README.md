# 第二章 环境准备

本章介绍了一系列环境配置的最佳实践。为了确保在后续章节中顺利使用 Jupyter Notebook, 强烈建议您按照以下相应步骤正确配置环境。

## 2.1 系统建议
首先，选择一个合适的系统。以下是推荐的硬件与操作系统列表：

>⚠️**硬件**

- 至少 16GB 内存的英特尔®个人电脑
- 搭载英特尔®至强®处理器和至少 32GB 内存的服务器

>⚠️**操作系统**

- Ubuntu 20.04 或更高版本
- CentOS 7 或更高版本
- Windows 10/11, 有无WSL均可

## 2.2 设置 Python 环境

接下来，使用 Python 环境管理工具（推荐使用 [Conda](https://docs.conda.io/projects/conda/en/stable/) ）创建 Python 环境并安装必要的库。


### 2.2.1 安装 Conda
请按照下面与您的操作系统相对应的说明进行操作。

#### 2.2.1.1 Linux

对于 Linux 用户，打开终端并且运行以下命令。

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda init
```
>**注意**
> 请按照终端显示的说明进行操作，直到 conda 初始化成功完成。


#### 2.2.1.2 Windows

对于 Windows 用户，在[这里](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)下载 conda 安装包并运行。

在安装完成后，打开 "Anaconda Powershell Prompt (Miniconda3)" 执行以下步骤。

#### 2.2.1.3 适用于 Linux 的 Windows 子系统 (WSL):

对于 WSL 用户，请确保已经安装了 WSL2。如果没有，请参阅[此处](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/win.html#install-wsl2l)了解安装方法。

打开 WSL2 shell 并运行与 [2.2.1.1 Linux](#2211-linux) 相同的命令。



### 2.2.2 创建环境
> **注意**
> 推荐使用 Python 3.9 运行 IPEX-LLM.

创建一个 Python 3.9 环境，名称由您选择，例如 `llm-tutorial`:
```
conda create -n llm-tutorial python=3.9
```
然后激活环境 `llm-tutorial`:
```
conda activate llm-tutorial
```

## 2.3 安装 IPEX-LLM

下面这一行命令将安装最新版本的`ipex-llm`以及所有常见LLM应用程序开发所需的依赖项。
```
pip install --pre --upgrade ipex-llm[all]
```

## 2.4 安装 Jupyter 服务

### 2.4.1 安装 Jupyter
运行教程提供的 Notebook (即 `.ipynb` 文件) 需要 `jupyter` 库。在激活的 Python 3.9 环境下运行：
```
pip install jupyter
```

### 2.4.2 启动 Jupyter 服务
启动 jupyter 服务的推荐指令在个人电脑和服务器上略有不同。

#### 2.4.2.1 在个人电脑上
在个人电脑上，只需在 shell 中运行以下命令：
```
jupyter notebook
```

#### 2.4.2.2 在服务器上
在服务器上，建议使用单个插槽的所有物理核心以获得更好的性能。因此，请运行以下命令：
```bash
# 以每个插槽有48个核心的服务器为例
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 jupyter notebook
```

祝贺您！现在您可以使用浏览器来访问 jupyter 服务 url 并运行本教程提供的notebooks。


## 2.5 关于使用LLM的一些你可能想要了解的事项

如果您在LLM和LLM应用程序开发方面是新手，本节可能包含了一些您想要了解的内容。

### 2.5.1 去哪里找模型

首先，您需要获取一个模型。社区中有许多开源的LLM可供选择。如果您没有特定的目标，可以考虑从社区公开的LLM排行榜上排名较高的模型中选择。这些公开的LLM排行榜一般采用多种评测手段评估和比较多个LLM的能力。一些比较有名的排行榜包括：

- [Open LLM LeaderBoard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 由 Huggingface 维护 
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 由 llmsys 维护

这些排行榜大多包含了列出的模型的参考链接。如果一个模型是开源的，您可以直接从提供的链接中轻松下载并尝试使用。


### 2.5.2 从Huggingface下载模型

截止到目前为止，许多热门的LLM模型都托管在Huggingface上。Huggingface托管的一个示例模型主页如下所示。
![image](https://github.com/shane-huang/bigdl-llm-tutorial/assets/1995599/a04df95f-5590-4bf1-968c-32cf494ece92)

要从Huggingface下载模型，您可以使用git或Huggingface提供的API。有关如何下载模型的详细信息，请参阅[从Huggingface下载模型](https://huggingface.co/docs/hub/models-downloading) 。

通常从Huggingface下载的模型可以使用[Huggingface Transformers库](https://huggingface.co/docs/transformers/index)加载。IPEX-LLM提供了API，可以轻松地与这些模型一起使用。请阅读本教程后续章节了解更多信息。
