from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def export_browser_model(run_dir: Path, output_dir: Path, sequence_length: int) -> Path:
    metadata = json.loads(
        (run_dir / "benchmark_results.json").read_text(encoding="utf-8")
    )
    model_type = metadata["selected_model"]
    if model_type != "transformer":
        raise ValueError(
            f"Browser ONNX export expects the selected transformer, got {model_type!r}"
        )
    model_dir = Path(metadata["models"][model_type]["model_path"]).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Selected transformer directory does not exist: {model_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new path")
    cli = shutil.which("optimum-cli")
    if not cli:
        raise RuntimeError("Install the browser export dependencies from requirements.browser.txt")

    output_dir.mkdir(parents=True)
    fp32_dir = output_dir / "onnx-fp32"
    q8_dir = output_dir / "onnx-int8-per-channel"
    subprocess.run(
        [
            cli,
            "export",
            "onnx",
            "--model",
            str(model_dir),
            "--task",
            "text-classification",
            "--opset",
            "17",
            "--sequence_length",
            str(sequence_length),
            "--batch_size",
            "1",
            str(fp32_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            cli,
            "onnxruntime",
            "quantize",
            "--onnx_model",
            str(fp32_dir),
            "-o",
            str(q8_dir),
            "--avx2",
            "--per_channel",
        ],
        check=True,
    )
    for filename in (
        "config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        shutil.copy2(fp32_dir / filename, q8_dir / filename)
    print(f"FP32 ONNX: {fp32_dir / 'model.onnx'}")
    print(f"Browser INT8 ONNX: {q8_dir / 'model_quantized.onnx'}")
    return q8_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the selected Amharic sentiment transformer for browser inference"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=72)
    args = parser.parse_args()
    export_browser_model(args.run_dir, args.output_dir, args.sequence_length)


if __name__ == "__main__":
    main()
