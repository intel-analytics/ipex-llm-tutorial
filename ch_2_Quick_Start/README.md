# Chapter 2 Quick Start

This chapter offers a step-by-step tutorial that allows for hands-on learning. We will begin by setting up required environment and then proceed to develop an application with BigDL-LLM transformers INT4 optimization. This application will allow us to conduct inferences on a large language model with low latency. By following this tutorial, you will gain a seamless experience that will enable you to easily comprehend and follow the upcoming tutorials.

## 1. Environment Setup

### 1.1 Recommended System

For a smooth experience, we recommend running the tutorial on PCs equipped with 12th Gen Intel® Core™ processor or higher, and at least 16GB RAM. For server users, we recommend the ones with Intel® Xeon® processors.

For OS, BigDL-LLM supports Ubuntu 20.04 or later, CentOS 7 or later, and Windows 10/11.

### 1.2 Conda and Environment Management

[Conda](https://docs.conda.io/projects/conda/en/stable/) is an open-source package & environment management system which is supported in multiple platforms. It provides a convenient way to manage packages and create isolated environments for different projects. We highly recommend using Conda here to create environment for the tutorials.

#### 1.2.1 Install Conda

##### 1.2.1.1 Linux
For Linux users, you could install Conda through:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```

Then you could run:
```bash
conda init
```
and follow the output instructions to finish the Conda initialization.


##### 1.2.1.2 Native Windows
For native Windows users, you could download Conda installer [here](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) based on your system information.

After the installation, open "Anaconda Powershell Prompt (Miniconda3)" for the following steps.

##### 1.2.1.3 Windows with WSL
For WSL users, you could follow the same instructions in section [1.2.1.1 Linux](#1211-linux).

> **Related Readings**
>
> For how to install WSL on your windows, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/win.html#install-wsl2) for more information.

#### 1.2.2 Create Environment
We suggest using Python 3.9 for BigDL-LLM. To create an environment with Python 3.9, run:
```
conda create -n llm-tutorial python=3.9
```
You have the flexibility to choose any name you prefer instead of `llm-tutorial`.

You can then activate the environment through:
```
conda activate llm-tutorial
```
and proceed with the installation of other packages.

### 1.3 Launch Tutorial
Package `jupyter` is required to be installed for running the tutorial notebooks. Under your activated Python 3.9 environment, run:
```
pip install jupyter
```

#### 1.3.1 Client
After installation, you could just use the following command on **client** machine:
```
jupyter notebook
```
to open and run the tutorial notebook [quick_start.ipynb](./quick_start.ipynb) in web browser.

#### 1.3.2 Server
For server users, it is recommended to run the tutorial with all the physical cores of a single socket. Run:
```bash
# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 jupyter notebook
```
to open and execute the tutorial notebook [quick_start.ipynb](./quick_start.ipynb) in web browser.