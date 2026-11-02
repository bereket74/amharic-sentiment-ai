from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amharic_sentiment.data import LABEL_NAMES, load_and_audit_splits
from amharic_sentiment.metrics import classification_metrics


def _predict(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[int]:
    predictions: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).tolist())
    return predictions


def evaluate(
    run_dir: Path,
    raw_dir: Path,
    quantized_model_dir: Path,
    output_path: Path,
    max_length: int = 72,
    batch_size: int = 24,
) -> dict[str, Any]:
    metadata = json.loads((run_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    if metadata["selected_model"] != "transformer":
        raise ValueError("The browser quantization evaluation requires the selected transformer.")
    model_dir = Path(metadata["models"]["transformer"]["model_path"])
    paths = {
        "train": raw_dir / "train.tsv",
        "validation": raw_dir / "dev.tsv",
        "test": raw_dir / "test.tsv",
    }
    loaded, audit = load_and_audit_splits(paths)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=False)
    full_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    int8_model = ORTModelForSequenceClassification.from_pretrained(
        str(quantized_model_dir), file_name="model_quantized.onnx"
    )

    result: dict[str, Any] = {
        "artifact": "dynamic INT8, AVX2 per-channel",
        "selection_rule": "quantization configuration selected using validation only",
        "dataset_commit": audit["source_commit"],
        "max_length": max_length,
        "validation_metrics": {},
        "full_precision_validation_metrics": {},
        "test_metrics": {},
        "full_precision_test_metrics": {},
    }
    for split in ("validation", "test"):
        texts = loaded[split]["texts"]
        labels = loaded[split]["labels"]
        reference = _predict(full_model, tokenizer, texts, max_length, batch_size)
        quantized = _predict(int8_model, tokenizer, texts, max_length, batch_size)
        ref_metrics = classification_metrics(labels, reference, LABEL_NAMES)
        int8_metrics = classification_metrics(labels, quantized, LABEL_NAMES)
        if split == "validation":
            result["full_precision_validation_metrics"] = ref_metrics
            result["validation_metrics"] = int8_metrics
            result["validation_label_agreement_with_full_model"] = float(
                np.mean(np.asarray(reference) == np.asarray(quantized))
            )
        else:
            result["full_precision_test_metrics"] = ref_metrics
            result["test_metrics"] = int8_metrics
            result["test_label_agreement_with_full_model"] = float(
                np.mean(np.asarray(reference) == np.asarray(quantized))
            )
            result["test_rows"] = len(texts)
    artifact = quantized_model_dir / "model_quantized.onnx"
    result["artifact_bytes"] = artifact.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Aggregate evaluation written to {output_path}")
    print(
        "Validation macro-F1: "
        f"full={result['full_precision_validation_metrics']['macro_f1']:.3f}, "
        f"INT8={result['validation_metrics']['macro_f1']:.3f}; "
        f"agreement={result['validation_label_agreement_with_full_model']:.1%}"
    )
    print(
        "Test macro-F1: "
        f"full={result['full_precision_test_metrics']['macro_f1']:.3f}, "
        f"INT8={result['test_metrics']['macro_f1']:.3f}; "
        f"agreement={result['test_label_agreement_with_full_model']:.1%}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the browser INT8 model without writing benchmark texts"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--quantized-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    evaluate(
        args.run_dir,
        args.raw_dir,
        args.quantized_model_dir,
        args.output,
        args.max_length,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
