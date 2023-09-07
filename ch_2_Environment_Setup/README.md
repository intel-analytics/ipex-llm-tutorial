# 第二章 配置环境

本章介绍了一系列环境配置的最佳实践。为了确保在后续章节中顺利使用 Jupyter Notebook, 强烈建议您按照以下相应步骤正确配置环境。

## 2.1 系统建议
首先，选择一个合适的系统。以下是推荐的硬件与操作系统列表：

>⚠️**硬件**

- 搭载第 12 代英特尔®酷睿™或更高版本的处理器和至少 16GB 内存的个人电脑
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

对于 WSL 用户，请确保已经安装了 WSL2. 如果没有，请参阅[此处](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/win.html#install-wsl2l) for how to install.
了解安装方法。

打开 WSL2 shell 并运行与 [2.2.1.1 Linux](#2211-linux) 相同的命令。



### 2.2.2 创建环境
> **注意**
> 推荐使用 Python 3.9 运行 BigDL-LLM.

创建一个 Python 3.9 环境，名称由您选择，例如 `llm-tutorial`:
```
conda create -n llm-tutorial python=3.9
```
然后激活环境 `llm-tutorial`:
```
conda activate llm-tutorial
```
## 2.3 安装 Jupyter 服务

### 2.3.1 安装 Jupyter
运行教程提供的笔记本 (即 `.ipynb` 文件) 需要 `jupyter` 库。在激活的 Python 3.9 环境下运行：
```
pip install jupyter
```

### 2.3.2 启动 Jupyter 服务
启动 jupyter 服务的推荐指令在个人电脑和服务器上略有不同。

#### 2.3.2.1 在个人电脑上
在个人电脑上，只需在 shell 中运行以下命令：
```
jupyter notebook
```

#### 2.3.2.2 在服务器上
在服务器上，建议使用单个插槽的所有物理核心以获得更好的性能。因此，请运行以下命令：
```bash
# 以每个插槽有48个核心的服务器为例
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 jupyter notebook
```

祝贺您！现在您可以使用浏览器来访问 jupyter 服务 url 并运行本教程提供的笔记本。
