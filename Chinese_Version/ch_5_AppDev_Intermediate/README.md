# 第五章 中级应用开发

您可以使用 BigDL-LLM 加载任何 Hugging Face *transformers* 模型，并在笔记本电脑上对其进行加速。有了 BigDL-LLM，托管在 Hugging Face 上的 PyTorch 模型（FP16/BF16/FP32 格式）可以通过低位量化（支持的精度包括 INT4/INT5/INT8）自动加载和优化。

本章将深入探讨 BigDL-LLM 的 `transformers` 样式 API，该 API 用于加载和优化 Huggingface *transformers* 模型。您将了解 API 的用法和常见做法，并学习如何使用这些 API 创建真实世界中的应用程序。

本章包含两个笔记本。

在笔记本 [4.1 运行 Transformer 模型](./4_1_Run_Transformer_Models.ipynb) 中，您将首先学习如何在不同场景中使用 `transformers` 样式的 API（例如保存/加载、精度选择等），然后继续构建一个具有流式显示和多轮聊天功能的聊天机器人应用程序。

在笔记本 [4.2 语音识别](./4_2_Speech_Recognition.ipynb) 中，您将学习如何使用 BigDL-LLM 加载基于 Transformer 的语音识别模型 [Whisper](https://openai.com/research/whisper)，然后使用它转录和翻译音频文件。
