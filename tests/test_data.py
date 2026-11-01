from __future__ import annotations

import pandas as pd

from amharic_sentiment.data import canonical_text, load_and_audit_splits


def _write_split(path, rows):
    pd.DataFrame(rows, columns=["tweet", "label"]).to_csv(path, sep="\t", index=False)
    return path


def test_canonical_text_only_normalizes_unicode_case_and_whitespace():
    assert canonical_text("  ሰላም\u00a0 WORLD 😊  ") == "ሰላም world 😊"
    assert canonical_text("አዎ!!!") != canonical_text("አዎ")


def test_overlap_guard_removes_training_and_validation_leakage(tmp_path):
    paths = {
        "train": _write_split(
            tmp_path / "train.tsv",
            [
                ("train-only", "positive"),
                ("leaks to validation", "negative"),
                ("duplicate train", "neutral"),
                ("duplicate train", "neutral"),
                ("label conflict", "positive"),
            ],
        ),
        "validation": _write_split(
            tmp_path / "dev.tsv",
            [
                ("leaks to validation", "negative"),
                ("validation-only", "neutral"),
                ("leaks to test", "positive"),
            ],
        ),
        "test": _write_split(
            tmp_path / "test.tsv",
            [
                ("leaks to test", "negative"),
                ("label conflict", "negative"),
                ("test-only", "positive"),
            ],
        ),
    }

    splits, audit = load_and_audit_splits(paths)

    assert "leaks to validation" not in splits["train"]["texts"]
    assert "label conflict" not in splits["train"]["texts"]
    assert splits["train"]["texts"].count("duplicate train") == 1
    assert "leaks to test" not in splits["validation"]["texts"]
    assert len(splits["test"]["texts"]) == 3
    assert audit["cross_split_train_rows_removed"] == 2
    assert audit["cross_split_validation_rows_removed"] == 1
    assert audit["unique_texts_with_conflicting_labels_across_original_splits"] == 2
    assert audit["test_split_modified"] is False


def test_label_ids_keep_documented_order(tmp_path):
    paths = {
        "train": _write_split(tmp_path / "train.tsv", [("ሰላም", "positive")]),
        "validation": _write_split(tmp_path / "dev.tsv", [("መልካም", "neutral")]),
        "test": _write_split(tmp_path / "test.tsv", [("አይ", "negative")]),
    }
    splits, _ = load_and_audit_splits(paths)
    assert splits["train"]["labels"] == [0]
    assert splits["validation"]["labels"] == [1]
    assert splits["test"]["labels"] == [2]
