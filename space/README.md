---
title: Amharic Sentiment AI
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---
# Amharic Sentiment AI — browser demo

This free Static Space runs the quantized AfroXLM-R-mini classifier in the visitor's browser with Transformers.js and ONNX Runtime Web. It does not use a hosted inference service or require a token.

On first use, the browser downloads roughly 141 MB of public model and tokenizer files from [`klohdfhk/amharic-sentiment-ai`](https://huggingface.co/klohdfhk/amharic-sentiment-ai). The model runs on the visitor's device; text entered in the demo is not sent to an inference server. Browser memory and speed vary by device.

The ONNX INT8 per-channel artifact is 118.5 MB. It was selected using validation only: macro-F1 0.659 versus 0.666 for the full-precision transformer. On the official 1,999-row test split, the quantized version scored 0.442 accuracy and 0.456 macro-F1 versus 0.460 and 0.466 for the full model; label agreement was 94.5%. See the model card and `browser_model_evaluation.json` for per-class results.

The test split has a substantial label-distribution shift from train and validation. Predictions are not calibrated probabilities, and the benchmark does not cover every dialect, domain, sarcasm, mixed-language post, or changing context. Do not use the model for high-impact decisions or profiling. No source dataset posts are included in this Space.

Source code: [bereket74/amharic-sentiment-ai](https://github.com/bereket74/amharic-sentiment-ai).
