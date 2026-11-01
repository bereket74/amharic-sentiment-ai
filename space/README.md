---
title: Amharic Sentiment AI
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
---

# Amharic Sentiment AI demo

Interactive demo for classifying short Amharic social-media-style messages as positive, neutral, or negative. The model artifact is loaded from `klohdfhk/amharic-sentiment-ai`; `model_config.json` selects the artifact loader.

Predictions are not calibrated probabilities. This low-resource benchmark does not cover every dialect, domain, sarcasm, mixed-language post, or rapidly changing context. Do not use the model for high-impact decisions or profiling.
