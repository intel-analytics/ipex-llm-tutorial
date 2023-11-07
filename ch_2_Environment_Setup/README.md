# Chapter 2 Environment Setup

This chapter presents a set of best practices for setting up your environment. To ensure a smooth experience with the notebooks in the subsequent chapters, it is strongly recommended that you follow the corresponding steps below to configure your environment properly.  

## 2.1 System Recommendation
First of all, choose a proper system. Here's a list of recommended hardware and OS.
>⚠️**Hardware**

- PCs equipped with 12th Gen Intel® Core™ processor or higher, and at least 16GB RAM
- Servers equipped with Intel® Xeon® processors, at least 32G RAM.

>⚠️**Operating System**

- Ubuntu 20.04 or later
- CentOS 7 or later
- Windows 10/11, with or without WSL

## 2.2 Setup Python Environment

Next, use a python environment management tool (we recommend using [Conda](https://docs.conda.io/projects/conda/en/stable/)) to create a python enviroment and install necessary libs.  


### 2.2.1 Install Conda
Follow the instructions corresponding to your OS below.

#### 2.2.1.1 Linux

For Linux users, open a terminal and run below commands.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda init
```
>**Note**
> Follow the instructions popped up on the console until conda initialization finished successfully.


#### 2.2.1.2 Windows

For Windows users, download conda installer [here](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) and execute it.

After the installation finished, open "Anaconda Powershell Prompt (Miniconda3)" for following steps.

#### 2.2.1.3 Windows Subsystem for Linux (WSL):

For WSL users, ensure you have already installed WSL2. If not, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/win.html#install-wsl2l) for how to install.

Open a WSL2 shell and run the same commands as in [2.2.1.1 Linux](#2211-linux) section.



### 2.2.2 Create Environment
> **Note**
> Python 3.9 is recommended for running BigDL-LLM.

Create a Python 3.9 environment with the name you choose, for example `llm-tutorial`:
```
conda create -n llm-tutorial python=3.9
```
Then activate the environment `llm-tutorial`:
```
conda activate llm-tutorial
```

## 2.3 Install BigDL-LLM

The one-line command below will install the latest `bigdl-llm` with all the dependencies for common LLM application development.
```
pip install --pre --upgrade bigdl-llm[all]
```

## 2.4 Setup Jupyter Service

### 2.4.1 Install Jupyter
The `jupyter` library is required for running the tutorial notebooks (i.e. the `.ipynb` files). Under your activated Python 3.9 environment, run:
```
pip install jupyter
```

### 2.4.2 Start Jupyter Service
The recommended command to start jupyter service is slightly different on PC and server. 

#### 2.4.2.1 On PC
On PC, just run the command in shell:
```
jupyter notebook
```

#### 2.4.2.2 On Server
On server, it is recommended to use all physical cores of a single socket for better performance. So run below command instead:
```bash
# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 jupyter notebook
```

Congratulations! Now you can use a web browser to access the jupyter service url and execute the notebooks provided in this tutorial. 


## 2.5 Things you may want to know about working with LLMs
If you're new to LLMs and LLM applicaiton development, there's something you might want to know. 

### 2.5.1 Where to find the models
To start, you'll need to obtain a model. There are numerous open-source LLMs available in the community. If you don't have a specific target in mind, consider selecting one that ranks higher on LLM leaderboards. These leaderboards evaluate and compare the capabilities of various LLMs. For instance,

- [Open LLM LeaderBoard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) hosted by Huggingface. 
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) hosted by llmsys.

Most of these leaderboards include reference links to the models listed. If a model is open source, you can easily download it directly from the provided link and give it a try.

### 2.5.2 Download Models from Huggingface
As of writing, many popular LLMs are hosted on [Huggingface](https://huggingface.co/). 
An example model homepage hosted on huggingface looks like this.

![image](https://github.com/shane-huang/bigdl-llm-tutorial/assets/1995599/a04df95f-5590-4bf1-968c-32cf494ece92)


To download models from huggingface, you can either use git or huggingface provided APIs. Refer to [Download Model from Huggingface](https://huggingface.co/docs/hub/models-downloading) for details about how to download models. 

Usually, the models downloaded from Huggingface can be loaded using [Huggingface Transformers library API](https://huggingface.co/docs/transformers/index). BigDL-LLM provides APIs to easily work with such models. Read the following chapters to to find out more. 
 


