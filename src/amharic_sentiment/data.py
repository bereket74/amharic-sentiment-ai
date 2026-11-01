from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DATASET_REPO = "afrisenti-semeval/afrisent-semeval-2023"
DATASET_COMMIT = "5aec3cfcf87b59f885e98641f72e52182776429d"
DATASET_URL = f"https://github.com/{DATASET_REPO}/tree/{DATASET_COMMIT}"
LABEL_NAMES = ["positive", "neutral", "negative"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def canonical_text(text: str) -> str:
    """Conservative exact-duplicate key; preserves words, punctuation, and emojis."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def download_dataset(raw_dir: str | Path) -> dict[str, Path]:
    """Download immutable official train/dev/test TSVs from the pinned upstream commit."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    session = requests.Session()
    session.headers["User-Agent"] = "amharic-sentiment-ai/0.1 (reproducible benchmark)"
    for split, filename in SPLIT_FILES.items():
        path = raw_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            url = (
                f"https://raw.githubusercontent.com/{DATASET_REPO}/"
                f"{DATASET_COMMIT}/data/amh/{filename}"
            )
            response = session.get(url, timeout=60)
            response.raise_for_status()
            if not response.content.startswith(b"tweet\tlabel"):
                raise ValueError(f"Unexpected AfriSenti TSV format for {split}: {url}")
            path.write_bytes(response.content)
        paths[split] = path
    return paths


def _read_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = {"tweet", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"AfriSenti split missing required columns: {sorted(missing)}")
    df = df[["tweet", "label"]].rename(columns={"tweet": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.casefold()
    df = df[df["text"].ne("") & df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    df["_key"] = df["text"].map(canonical_text)
    return df.reset_index(drop=True)


def load_and_audit_splits(
    paths: dict[str, str | Path],
) -> tuple[dict[str, dict[str, list[Any]]], dict[str, Any]]:
    """Load official splits and remove only train/validation records that overlap later splits.

    The official test set is preserved byte-for-byte at the row level. Validation examples
    that also occur in test are dropped; training examples overlapping either later split
    are dropped. Repeated examples within train/validation keep their first occurrence.
    """
    frames = {name: _read_split(path) for name, path in paths.items()}
    source_files = {}
    for name, path in paths.items():
        payload = Path(path).read_bytes()
        source_files[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    raw_counts = {name: int(len(frame)) for name, frame in frames.items()}
    raw_label_counts = {
        name: {label: int((frame["label"] == label).sum()) for label in LABEL_NAMES}
        for name, frame in frames.items()
    }

    test_keys = set(frames["test"]["_key"])
    val_keys = set(frames["validation"]["_key"])
    raw_train = frames["train"]
    raw_val = frames["validation"]

    train_cross = raw_train["_key"].isin(test_keys | val_keys)
    val_cross = raw_val["_key"].isin(test_keys)
    cross_split_train_removed = int(train_cross.sum())
    cross_split_val_removed = int(val_cross.sum())
    train_kept = raw_train.loc[~train_cross].drop_duplicates("_key", keep="first")
    val_kept = raw_val.loc[~val_cross].drop_duplicates("_key", keep="first")
    test_kept = frames["test"].copy()  # do not edit official test rows

    # Audit cross-split label conflicts using original data, before overlap removal.
    all_rows = pd.concat(
        [frame.assign(_split=name) for name, frame in frames.items()], ignore_index=True
    )
    label_sets = all_rows.groupby("_key")["label_id"].nunique()
    conflicting_texts = int((label_sets > 1).sum())
    duplicates_within = {
        name: int(len(frame) - frame["_key"].nunique()) for name, frame in frames.items()
    }
    splits = {"train": train_kept, "validation": val_kept, "test": test_kept}
    loaded: dict[str, dict[str, list[Any]]] = {}
    final_counts: dict[str, int] = {}
    final_label_counts: dict[str, dict[str, int]] = {}
    for name, frame in splits.items():
        final_counts[name] = int(len(frame))
        final_label_counts[name] = {
            label: int((frame["label"] == label).sum()) for label in LABEL_NAMES
        }
        loaded[name] = {
            "texts": frame["text"].tolist(),
            "labels": frame["label_id"].astype(int).tolist(),
            "label_strings": frame["label"].tolist(),
        }

    # A remaining exact text overlap indicates a bug in the leakage guard.
    train_keys = set(train_kept["_key"])
    val_keys_final = set(val_kept["_key"])
    if train_keys & val_keys_final or train_keys & test_keys or val_keys_final & test_keys:
        raise AssertionError("Exact text overlap remains across train/validation/test after audit")

    audit = {
        "dataset": "AfriSenti Amharic (official SemEval 2023 Task 12 release)",
        "source_repository": DATASET_REPO,
        "source_commit": DATASET_COMMIT,
        "source_url": DATASET_URL,
        "source_files": source_files,
        "license": "CC BY 4.0",
        "label_names_and_ids": LABEL2ID,
        "raw_split_rows": raw_counts,
        "raw_label_counts": raw_label_counts,
        "final_split_rows": final_counts,
        "final_label_counts": final_label_counts,
        "exact_duplicate_rows_within_original_splits": duplicates_within,
        "cross_split_train_rows_removed": cross_split_train_removed,
        "cross_split_validation_rows_removed": cross_split_val_removed,
        "unique_texts_with_conflicting_labels_across_original_splits": conflicting_texts,
        "test_split_modified": False,
        "duplicate_key_policy": (
            "NFKC normalization, Unicode casefolding, and whitespace collapse; "
            "punctuation/emojis retained"
        ),
    }
    return loaded, audit


def save_audit(audit: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
