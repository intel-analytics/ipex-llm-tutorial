# Chapter 5 Application Development: Intermediate

You can use BigDL-LLM to load any Hugging Face *transformers* model and accelerate it on your laptop. With BigDL-LLM, PyTorch models (in FP16/BF16/FP32) hosted on Hugging Face can be loaded and optimized automatically with low-bit quantizations (supported precisions include INT4/INT5/INT8).

This chapter is a deeper dive of the BigDL-LLM `transformers`-style API, which is used to load and optimize Huggingface *transformers* models. You'll learn about the API usage and common practices, and learn how to create real-world applications using these APIs.

Two notebooks are included in this chapter. 

In the notebook [4_1_Run_Transformer_Models](./4_1_Run_Transformer_Models.ipynb), you'll first learn how to use `transformers`-style API in different scenarios (e.g. save/load, precision choices, etc.), then proceed to build a chatbot application with streaming and multi-turn chat capabilities.

In the notebook [4_2_Speech_Recognition](./4_2_Speech_Recognition.ipynb), you'll learn how to use BigDL-LLM to load a Transformer-based speech recognition model [Whisper](https://openai.com/research/whisper), and then use it to transcribe and translate audio files.
