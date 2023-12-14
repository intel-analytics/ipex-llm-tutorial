# Chapter 6 GPU Acceleration

Apart from the significant acceleration capabilites on Intel CPUs, BigDL-LLM also supports optimizations and acceleration for running LLMs (large language models) on Intel GPUs.

BigDL-LLM supports optimizations of any [*HuggingFace transformers*](https://huggingface.co/docs/transformers/index) model on Intel GPUs with the help of low-bit techniques, modern hardware accelerations and latest software optimizations.

#### 6B model running on Intel Arc GPU (real-time screen capture):

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/chatglm2-arc.gif" width='60%' /> 

</p>

#### 13B model running on Intel Arc GPU (real-time screen capture): 

<p align="left">
            <img src="https://llm-assets.readthedocs.io/en/latest/_images/llama2-13b-arc.gif" width='60%' /> 

</p>

In Chapter 6, you will learn how to run LLMs, as well as implement stream chat functionalities, using BigDL-LLM optimizations on Intel GPUs. Popular open source models are used as examples:

+ [Llama2-7B](./6_1_GPU_Llama2-7B.md)

## 6.0 Environment Setup

Here are some best practices for setting up your environment. It is strongly recommended that you follow the corresponding steps below to configure your environment properly.

### 6.0.1 System Recommendation

For a smooth experience with the notebooks in Chatper 7, please ensure your hardware and OS meet the following requirements:

> ⚠️Hardware
  - Intel Arc™ A-Series Graphics
  - Intel Data Center GPU Flex Series

> ⚠️Operating System
  - Linux system, Ubuntu 22.04 is preferred

    > **Note**
    > Please note that only Linux OS has been supported for BigDL-LLM optimizations on Intel GPUs.

### 6.0.2 Driver and Toolkit Installation

Before benifiting from BigDL-LLM on Intel GPUs, there’re several steps for tools installation:

- First you need to install Intel GPU driver. Please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
  > **Note**
  > For BigDL-LLM with default IPEX version (IPEX 2.0.110+xpu), Intel GPU Driver version [Stable 647.21](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html) is required.

- You also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
  > **Note**
  > BigDL-LLM with default IPEX version (IPEX 2.0.110+xpu) requires Intel® oneAPI Base Toolkit's version == 2023.2.0.

<details><summary>For client users with Intel Arc™ A-Series Graphics on Unbuntu 22.04, you could also refer to the commands below for driver and oneAPI Base Toolkit installation. Show detailed commands:</summary>
<br/>

```bash
# Install Arc driver 
sudo apt-get install -y gpg-agent wget

wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
  sudo tee  /etc/apt/sources.list.d/intel.gpu.jammy.list


# Downgrade kernel
sudo apt-get update && sudo apt-get install  -y --install-suggests  linux-image-5.19.0-41-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no '5.19.0-41' | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo  update-grub

sudo reboot

# Remove latest kernel
sudo apt purge linux-image-6.2.0-26-generic

sudo apt autoremove

sudo reboot

# Install drivers
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

# Configuring permissions
sudo gpasswd -a ${USER} render

newgrp render

# Verify the device is working with i915 driver
sudo apt-get install -y hwinfo
hwinfo --display


# Install one api
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install intel-basekit
```
</details>

### 6.0.3 Python Environment Setup

Next, use a python environment management tool (we recommend using [Conda](https://docs.conda.io/projects/conda/en/stable/)) to create a python enviroment and install necessary libs.

#### 6.0.3.1 Install Conda

For Linux users, open a terminal and run below commands:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda init
```

> **Note**
> Follow the instructions popped up on the console until conda initialization finished successfully.

#### 6.0.3.2 Create Environment

> **Note**
> Python 3.9 is recommended for running BigDL-LLM.

Create a Python 3.9 environment with the name you choose, for example `llm-tutorial-gpu`:

```bash
conda create -n llm-tutorial-gpu python=3.9
```

Then activate the environment `llm-tutorial-gpu`:

```bash
conda activate llm-tutorial-gpu
```

### 6.0.4 Best Known Configuration on Linux

For optimal performance on Intel GPUs, it is recommended to set several environment variables:

```bash
# configure OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
