# 6.2 Run Whisper (medium) on Intel GPUs

You can use IPEX-LLM to load Transformer-based automatic speech recognition (ASR) models for acceleration on Intel GPUs. With IPEX-LLM, PyTorch models (in FP16/BF16/FP32) for ASR can be loaded and optimized automatically on Intel GPUs with low-bit quantization (supported precisions include INT4/NF4/INT5/FP8/INT8).

In this tutorial, you will learn how to run speech models on Intel GPUs with IPEX-LLM optimizations, and based on that build a speech recognition application. A popular open-source model for both ASR and speech translation, [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) is used as an example.

> [!NOTE]
> Please make sure that you have prepared the environment for IPEX-LLM on GPU before you started. Refer to [here](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for more information regarding installation and environment preparation. Besides, to process audio files, you also need to install `librosa` by performing `pip install -U librosa`.

## 6.2.1 Download Audio Files
To start with, the first thing to do is preparing some audio files for this demo. As an example, you can download an English example from multilingual audio dataset [voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) and one Chinese example from the Chinese audio dataset [AIShell](https://huggingface.co/datasets/carlot/AIShell). You are free to pick other recording clips found within or outside the dataset. 


## 6.2.2 Load Model in Low Precision

One common use case is to load a model from Hugging Face with IPEX-LLM low-bit precision optimization. For Whisper (medium), you could simply import `ipex_llm.transformers.AutoModelForSpeechSeq2Seq` instead of `transformers.AutoModelForSpeechSeq2Seq`, and specify `load_in_4bit=True` parameter accordingly in the `from_pretrained` function.

For Intel GPUs, **once you have the model in low precision, set it to `to('xpu')`.**

**For INT4 Optimizations (with `load_in_4bit=True`):**

```python
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq

model_in_4bit = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-medium",
                                                  load_in_4bit=True)
model_in_4bit_gpu = model_in_4bit.to('xpu')
```

> [!NOTE]
> * IPEX-LLM has supported `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSpeechSeq2Seq` and `AutoModelForSeq2SeqLM`, etc.
>
>   If you have already downloaded the Whisper (medium) model, you could specify `pretrained_model_name_or_path` to the model path.
>
> * Currently, `load_in_low_bit` supports options `'sym_int4'`, `'asym_int4'`, `'sym_int8'`, `'nf4'`, `'fp6'`, `'fp8'`,`'fp16'`, etc., in which `'sym_int4'` means symmetric int 4, `'asym_int4'` means asymmetric int 4, and `'nf4'` means 4-bit NormalFloat, etc. Relevant low bit optimizations will be applied to the model.
>
>   `load_in_4bit=True` is equivalent to `load_in_low_bit='sym_int4'`.
>
> * When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
> 
>   This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
>
> * You could refer to the [API documentation](https://ipex-llm.readthedocs.io/en/latest/doc/PythonAPI/LLM/transformers.html) for more information.

## 6.2.3 Load Whisper Processor

A Whisper processor is also needed for both audio pre-processing, and post-processing model outputs from tokens to texts. IPEX-LLM does not provide a customized implementation for that, so you might want to use the official `transformers` API to load `WhisperProcessor`:

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="openai/whisper-medium")
```

> [!NOTE]
> If you have already downloaded the Whisper (medium) model, you could specify `pretrained_model_name_or_path` to the model path.

## 6.2.4 Run Model to Transcribe English Audio

Once you have optimized the Whisper model using IPEX-LLM with INT4 optimization and loaded the Whisper processor, you are ready to begin transcribing the audio through model inference.

Let's start with the English audio file `audio_en.wav`. Before we feed it into Whisper processor, we need to extract sequence data from raw speech waveform:

```python
import librosa

data_en, sample_rate_en = librosa.load("audio_en.wav", sr=16000)
```

> [!NOTE]
> For `whisper-medium`, its `WhisperFeatureExtractor` (part of `WhisperProcessor`) extracts features from audio using a 16,000Hz sampling rate by default. It's important to load the audio file at the sample sampling rate with model's `WhisperFeatureExtractor` for precise recognition.
> 

We can then proceed to transcribe the audio file based on the sequence data, using exactly the same way as using official `transformers` API:

```python
import torch
import time

# define task type
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

with torch.inference_mode():
    # extract input features for the Whisper model
    input_features = processor(data_en, sampling_rate=sample_rate_en, return_tensors="pt").input_features.to('xpu')

    # predict token ids for transcription
    predicted_ids = model_in_4bit_gpu.generate(input_features, forced_decoder_ids=forced_decoder_ids,max_new_tokens=200)

    # decode token ids into texts
    transcribe_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print('-'*20, 'English Transcription', '-'*20)
    print(transcribe_str)
```

> [!NOTE]
> `forced_decoder_ids` defines the context token for different language and task (transcribe or translate). If it is set to `None`, Whisper will automatically predict them.
> 


## 6.2.5 Run Model to Transcribe Chinese Audio and Translate to English

Next, let's move to the Chinese audio `audio_zh.wav`. Whisper offers capability to transcribe multilingual audio files, and translate the recognized text into English. The only difference here is to define specific context token through `forced_decoder_ids`:

```python
# extract sequence data
data_zh, sample_rate_zh = librosa.load("audio_zh.wav", sr=16000)

# define Chinese transcribe task
forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

with torch.inference_mode():
    input_features = processor(data_zh, sampling_rate=sample_rate_zh, return_tensors="pt").input_features.to('xpu')
    predicted_ids = model_in_4bit.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcribe_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print('-'*20, 'Chinese Transcription', '-'*20)
    print(transcribe_str)

# define Chinese transcribe and translation task
forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="translate")

with torch.inference_mode():
    input_features = processor(data_zh, sampling_rate=sample_rate_zh, return_tensors="pt").input_features.to('xpu')
    predicted_ids = model_in_4bit.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=200)
    translate_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print('-'*20, 'Chinese to English Translation', '-'*20)
    print(translate_str)
```
