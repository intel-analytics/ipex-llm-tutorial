# Chapter 5 Application Development: Intermediate

You can use IPEX-LLM to load any Hugging Face *transformers* model and accelerate it on your laptop. With IPEX-LLM, PyTorch models (in FP16/BF16/FP32) hosted on Hugging Face can be loaded and optimized automatically with low-bit quantizations (supported precisions include INT4/INT5/INT8).

This chapter is a deeper dive of the IPEX-LLM `transformers`-style API, which is used to load and optimize Huggingface *transformers* models. You'll learn about the API usage and common practices, and learn how to create real-world applications using these APIs.

Two notebooks are included in this chapter. 

In the notebook [5_1_ChatBot](./5_1_ChatBot.ipynb), you'll first learn how to use `transformers`-style API in different scenarios (e.g. save/load, precision choices, etc.), then proceed to build a chatbot application with streaming and multi-turn chat capabilities.

In the notebook [5_2_Speech_Recognition](./5_2_Speech_Recognition.ipynb), you'll learn how to use IPEX-LLM to load a Transformer-based speech recognition model [Whisper](https://openai.com/research/whisper), and then use it to transcribe and translate audio files.
