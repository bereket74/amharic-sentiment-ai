from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from amharic_sentiment.inference import SentimentPredictor  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "klohdfhk/amharic-sentiment-ai")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
MODEL_CONFIG = ROOT / "model_config.json"
if MODEL_CONFIG.is_file():
    _config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
else:
    _config = {"model_type": "transformer"}
MODEL_TYPE = os.environ.get("MODEL_TYPE", _config["model_type"])

if MODEL_TYPE == "transformer":
    predictor = SentimentPredictor("transformer", LOCAL_MODEL_PATH or MODEL_ID)
else:
    model_dir = (
        Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path(snapshot_download(repo_id=MODEL_ID))
    )
    model_filename = "model.joblib" if MODEL_TYPE == "classical" else "neural_weights.npz"
    model_file = model_dir / model_filename if model_dir.is_dir() else model_dir
    predictor = SentimentPredictor(MODEL_TYPE, model_file)


def classify(text: str) -> str:
    try:
        label = predictor.predict_label(text)
    except ValueError as exc:
        return f"Please check the text: {exc}"
    return (
        f"## {label.title()}\n\n"
        "This is a three-way sentiment prediction, not a measure of certainty."
    )


with gr.Blocks(title="Amharic Sentiment AI") as demo:
    gr.Markdown(
        "# Amharic Sentiment AI\n"
        "Enter a short Amharic message to classify its sentiment as **positive**, "
        "**neutral**, or **negative**."
    )
    text = gr.Textbox(
        label="Amharic text",
        placeholder="እውነት በጣም ጥሩ ነበር!",
        lines=4,
        max_lines=8,
        max_length=5000,
    )
    result = gr.Markdown(label="Prediction")
    button = gr.Button("Analyze sentiment", variant="primary")
    button.click(classify, inputs=text, outputs=result)
    text.submit(classify, inputs=text, outputs=result)
    gr.Markdown(
        "**Limitations:** This model was trained on the AfriSenti Amharic social-media "
        "benchmark. Sarcasm, dialect, mixed-language writing, and context can change the "
        "result. Do not use it for high-impact decisions or profiling."
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
