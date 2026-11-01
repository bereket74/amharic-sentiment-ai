from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from amharic_sentiment.inference import SentimentPredictor


def create_demo(predictor: SentimentPredictor) -> gr.Blocks:
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
            "Try a short Amharic social-media-style message. The model returns **positive**, "
            "**neutral**, or **negative**."
        )
        text = gr.Textbox(
            label="Amharic text",
            placeholder="እውነት በጣም ጥሩ ነበር!",
            lines=4,
            max_lines=8,
            max_length=5000,
        )
        output = gr.Markdown(label="Prediction")
        submit = gr.Button("Analyze sentiment", variant="primary")
        submit.click(classify, inputs=text, outputs=output)
        text.submit(classify, inputs=text, outputs=output)
        gr.Markdown(
            "**Limitations:** Trained on the AfriSenti Amharic social-media benchmark. "
            "Sarcasm, dialect, mixed-language writing, and context can change the result. "
            "Do not use this demo for high-impact decisions or profiling."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Amharic Sentiment AI demo")
    parser.add_argument(
        "--model-type", choices=["classical", "neural", "transformer"], required=True
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    predictor = SentimentPredictor(args.model_type, args.model_path)
    create_demo(predictor).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
