from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from amharic_sentiment.models.transformer import MODEL_NAME, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report token-length statistics from train.tsv only; never prints text."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/raw/afrisenti-amharic/train.tsv"),
    )
    args = parser.parse_args()
    if not args.train_file.is_file():
        raise FileNotFoundError(
            f"Training split not found: {args.train_file}. Run scripts/prepare_data.py first."
        )
    frame = pd.read_csv(args.train_file, sep="\t", dtype=str, keep_default_na=False)
    if "tweet" not in frame.columns:
        raise ValueError("Expected the official AfriSenti train.tsv 'tweet' column.")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, fix_mistral_regex=False
    )
    tokenized = tokenizer(frame["tweet"].tolist(), truncation=False, padding=False)
    lengths = np.asarray([len(row) for row in tokenized["input_ids"]], dtype=np.int64)
    result = {
        "split": "train only",
        "rows": int(len(lengths)),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "p50": float(np.quantile(lengths, 0.50)),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "max": int(lengths.max()),
        "prints_source_text": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
