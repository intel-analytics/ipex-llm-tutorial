# 第六章 GPU 支持

BigDL-LLM 除了在英特尔 CPU 上具有显著的加速能力外，还支持在英特尔 GPU 上运行 LLM（大型语言模型）的优化和加速。

BigDL-LLM 借助低精度技术、现代硬件加速和最新的软件优化，支持在英特尔 GPU 上优化任何 [*HuggingFace transformers*](https://huggingface.co/docs/transformers/index) 模型。

#### 在英特尔锐炫 GPU 上运行 6B 模型（实时屏幕画面）:

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif" width='60%' /> 

</p>

#### 在英特尔锐炫 GPU 上运行 13B 模型（实时屏幕画面）: 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif" width='60%' /> 

</p>

在第七章中，您将学习如何在英特尔 GPU 上使用 BigDL-LLM 优化来运行 LLM 以及实现流式对话功能。本章将使用流行的开源模型作为示例：

+ [Llama2-7B](./7_1_GPU_Llama2-7B.md)

## 6.0 环境配置

以下是一些设置环境的最佳做法。强烈建议您按照以下相应步骤正确配置环境。

### 6.0.1 系统需求

为了顺利体验第七章中的笔记本，请确保您的硬件和操作系统符合以下要求：

> ⚠️硬件
  - 英特尔锐炫™ A系列显卡
  - 英特尔 Data Center GPU Flex Series

> ⚠️操作系统
  - Linux 系统, 推荐使用 Ubuntu 22.04

    > **注意**
    > 请注意，英特尔 GPU 上的 BigDL-LLM 优化仅支持 Linux 操作系统。

### 6.0.2 安装驱动程序和工具包

在英特尔 GPU 上使用 BigDL-LLM 之前，有几个安装工具的步骤：

- 首先，您需要安装英特尔 GPU 驱动程序。请参阅我们的[驱动程序安装](https://dgpu-docs.intel.com/driver/installation.html)以了解更多关于通用 GPU 功能的事项。
  > **注意**
  > 对于使用默认 IPEX 版本（IPEX 2.0.110+xpu）的 BigDL-LLM，需要英特尔 GPU 驱动程序版本 [Stable 647.21](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html)。

- 您还需要下载并安装[英特尔® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)。OneMKL 和 DPC++ 编译器是必选项，其他为可选项。
  > **注意**
  > 使用默认 IPEX 版本（IPEX 2.0.110+xpu）的 BigDL-LLM 需要英特尔® oneAPI Base Toolkit 的版本 >= 2023.2.0。

<details><summary>对于在 Ubuntu 22.04 上使用英特尔锐炫™ A 系列显卡的客户端用户，也可参考以下命令安装驱动程序和 oneAPI Base Toolkit。详细命令：</summary>
<br/>

```bash
# 安装锐炫驱动程序
sudo apt-get install -y gpg-agent wget

wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
  sudo tee  /etc/apt/sources.list.d/intel.gpu.jammy.list


# 降级内核版本
sudo apt-get update && sudo apt-get install  -y --install-suggests  linux-image-5.19.0-41-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no '5.19.0-41' | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo  update-grub

sudo reboot

# 移除最新版本内核
sudo apt purge linux-image-6.2.0-26-generic

sudo apt autoremove

sudo reboot

# 安装驱动程序
sudo apt-get update

sudo apt-get -y install \
    gawk \
    dkms \
    linux-headers-$(uname -r) \
    libc6-dev
	
sudo apt-get install -y intel-platform-vsec-dkms intel-platform-cse-dkms intel-i915-dkms intel-fw-gpu

sudo apt-get install -y gawk libc6-dev udev\
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
  
sudo reboot

# 配置权限
sudo gpasswd -a ${USER} render

newgrp render

# 验证设备是否可以使用 i915 驱动程序正常运行
sudo apt-get install -y hwinfo
hwinfo --display


# 安装 one api
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install intel-basekit
```
</details>

### 7.0.3 Python 环境配置

接下来，使用 python 环境管理工具（推荐使用 [Conda](https://docs.conda.io/projects/conda/en/stable/)）创建 python 环境并安装必要的库。

#### 7.0.3.1 安装 Conda

对于 Linux 用户，打开终端并运行以下命令：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda init
```

> **注意**
> 按照控制台弹出的提示操作，直到 conda 初始化成功完成。

#### 6.0.3.2 创建环境

> **注意**
> 推荐使用 Python 3.9 运行 BigDL-LLM.

使用您选择的名称创建一个 Python 3.9 环境，例如 `llm-tutorial-gpu`：

```bash
conda create -n llm-tutorial-gpu python=3.9
```

接下来激活环境 `llm-tutorial-gpu`:

```bash
conda activate llm-tutorial-gpu
```

### 6.0.4 Linux 上的推荐配置

为优化英特尔 GPU 的性能，建议设置以下几个环境变量：

```bash
# 配置 OneAPI 环境变量
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
