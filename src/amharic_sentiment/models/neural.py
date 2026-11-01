"""A compact, from-scratch BiGRU baseline for Amharic sentiment classification.

Validation macro-F1 selects the training duration; a fresh model is then refit on
train+validation data, and the held-out test split is evaluated only afterward.
"""

from __future__ import annotations

import json
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from amharic_sentiment.metrics import classification_metrics

_PAD_TOKEN = "<pad>"
_UNK_TOKEN = "<unk>"
_PAD_ID = 0
_UNK_ID = 1
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The implementation is CPU-only, which avoids nondeterministic GPU kernels.
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def _tokenize(text: Any) -> list[str]:
    if text is None:
        return []
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _normalize_labels(values: Sequence[int], split_name: str, n_classes: int) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            label = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{split_name} labels must be integers in [0, {n_classes - 1}]."
            ) from exc
        if label != value or label < 0 or label >= n_classes:
            raise ValueError(f"{split_name} labels must be integers in [0, {n_classes - 1}].")
        result.append(label)
    return result


def _build_vocab(train_tokens: Sequence[Sequence[str]], max_vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter(token for row in train_tokens for token in row)
    ordered = sorted(counts, key=lambda token: (-counts[token], token))[: max_vocab_size - 2]
    vocab = {_PAD_TOKEN: _PAD_ID, _UNK_TOKEN: _UNK_ID}
    vocab.update({token: index for index, token in enumerate(ordered, start=2)})
    return vocab


def _encode(
    token_rows: Sequence[Sequence[str]], vocab: dict[str, int], max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((len(token_rows), max_length), _PAD_ID, dtype=np.int64)
    lengths = np.empty(len(token_rows), dtype=np.int64)
    for row_index, tokens in enumerate(token_rows):
        encoded = [vocab.get(token, _UNK_ID) for token in tokens[:max_length]]
        if not encoded:
            encoded = [_UNK_ID]
        length = min(len(encoded), max_length)
        ids[row_index, :length] = encoded[:length]
        lengths[row_index] = length
    return ids, lengths


class _BiGRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=_PAD_ID)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        # A single bidirectional layer returns the final forward and backward states.
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.output_dropout(features))


def _predict(
    model: nn.Module, encoded: np.ndarray, lengths: np.ndarray, batch_size: int
) -> list[int]:
    model.eval()
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            end = min(start + batch_size, len(encoded))
            batch_ids = torch.as_tensor(encoded[start:end], dtype=torch.long)
            batch_lengths = torch.as_tensor(lengths[start:end], dtype=torch.long)
            logits = model(batch_ids, batch_lengths)
            predictions.extend(logits.argmax(dim=1).tolist())
    return predictions


def run_experiment(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    output_dir: str | Path,
    label_names: Sequence[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Train, select on validation macro-F1, and evaluate a 3-class BiGRU.

    Labels are expected to follow the provided ``label_names`` order (for this
    task: positive=0, neutral=1, negative=2). Validation labels choose the epoch;
    train+validation labels fit the final model; test labels are used only for metrics.
    """
    names = list(label_names)
    n_classes = len(names)
    if n_classes != 3:
        raise ValueError(f"This baseline expects exactly 3 classes; received {n_classes}.")
    train_y = _normalize_labels(train_labels, "train", n_classes)
    val_y = _normalize_labels(val_labels, "validation", n_classes)
    train_texts = list(train_texts)
    val_texts = list(val_texts)
    if not train_texts or not val_texts or len(test_texts) == 0:
        raise ValueError("Train, validation, and test splits must each be non-empty.")
    if not (
        len(train_texts) == len(train_y)
        and len(val_texts) == len(val_y)
        and len(test_texts) == len(test_labels)
    ):
        raise ValueError("Each text split must have the same number of items as its label split.")

    _seed_everything(int(seed))

    # Fixed, modest architecture and bounded sequence lengths keep CPU cost low.
    config: dict[str, Any] = {
        "architecture": "word_embedding_bidirectional_gru",
        "embedding_dim": 64,
        "hidden_dim": 48,
        "dropout": 0.30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "max_epochs": 20,
        "early_stopping_patience": 4,
        "max_vocab_size": 30000,
        "max_sequence_length": 160,
        "loss": "train-frequency-balanced cross entropy",
    }

    train_tokens = [_tokenize(text) for text in train_texts]
    selection_vocab = _build_vocab(train_tokens, config["max_vocab_size"])
    # Selection-stage sequence length and vocabulary use training texts only.
    selection_lengths = [max(1, len(row)) for row in train_tokens]
    selection_percentile = int(np.percentile(selection_lengths, 95))
    selection_max_length = min(config["max_sequence_length"], max(16, selection_percentile))
    selection_train_x, selection_train_lens = _encode(
        train_tokens, selection_vocab, selection_max_length
    )
    selection_val_x, selection_val_lens = _encode(
        [_tokenize(text) for text in val_texts], selection_vocab, selection_max_length
    )

    selection_model = _BiGRUClassifier(
        vocab_size=len(selection_vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        n_classes=n_classes,
    )
    train_y_array = np.asarray(train_y, dtype=np.int64)
    class_counts = np.bincount(train_y_array, minlength=n_classes).astype(np.float32)
    present = class_counts > 0
    # Selection-stage weights are based on the original training subset only.
    selection_class_weights = np.zeros(n_classes, dtype=np.float32)
    if present.any():
        selection_class_weights[present] = len(train_y) / (
            int(present.sum()) * class_counts[present]
        )
    selection_loss = nn.CrossEntropyLoss(
        weight=torch.as_tensor(selection_class_weights, dtype=torch.float32)
    )
    selection_optimizer = torch.optim.AdamW(
        selection_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_f1 = -1.0
    best_epoch = 0
    best_val_predictions: list[int] | None = None
    epochs_without_improvement = 0
    training_started = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        selection_model.train()
        permutation = torch.randperm(len(train_y), generator=torch.default_generator)
        for start in range(0, len(train_y), config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]].numpy()
            batch_ids = torch.as_tensor(selection_train_x[indices], dtype=torch.long)
            batch_lengths = torch.as_tensor(selection_train_lens[indices], dtype=torch.long)
            batch_labels = torch.as_tensor(train_y_array[indices], dtype=torch.long)
            selection_optimizer.zero_grad(set_to_none=True)
            logits = selection_model(batch_ids, batch_lengths)
            loss = selection_loss(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(selection_model.parameters(), max_norm=1.0)
            selection_optimizer.step()

        val_predictions = _predict(
            selection_model, selection_val_x, selection_val_lens, config["batch_size"]
        )
        val_metrics_this_epoch = classification_metrics(val_y, val_predictions, names)
        current_f1 = float(val_metrics_this_epoch["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_val_predictions = val_predictions
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                break

    if best_epoch == 0 or best_val_predictions is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    val_predictions = best_val_predictions
    val_metrics = classification_metrics(val_y, val_predictions, names)

    # Rebuild both vocabulary and sequence length from combined train+validation
    # texts; test text is not inspected during selection or final fitting.
    combined_texts = train_texts + val_texts
    combined_y = train_y + val_y
    combined_tokens = [_tokenize(text) for text in combined_texts]
    vocab = _build_vocab(combined_tokens, config["max_vocab_size"])
    combined_lengths = [max(1, len(row)) for row in combined_tokens]
    combined_percentile = int(np.percentile(combined_lengths, 95))
    max_length = min(config["max_sequence_length"], max(16, combined_percentile))
    config["selection_sequence_length"] = selection_max_length
    config["sequence_length"] = max_length
    combined_x, combined_seq_lengths = _encode(combined_tokens, vocab, max_length)
    combined_y_array = np.asarray(combined_y, dtype=np.int64)

    combined_counts = np.bincount(combined_y_array, minlength=n_classes).astype(np.float32)
    combined_present = combined_counts > 0
    class_weights = np.zeros(n_classes, dtype=np.float32)
    if combined_present.any():
        class_weights[combined_present] = len(combined_y) / (
            int(combined_present.sum()) * combined_counts[combined_present]
        )
    final_loss = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32))
    # The new model and its fixed-epoch refit are reproducible independently
    # from the amount of work done during validation-based selection.
    _seed_everything(int(seed))
    combined_model = _BiGRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        n_classes=n_classes,
    )
    final_optimizer = torch.optim.AdamW(
        combined_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    for _ in range(best_epoch):
        combined_model.train()
        permutation = torch.randperm(len(combined_y), generator=torch.default_generator)
        for start in range(0, len(combined_y), config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]].numpy()
            batch_ids = torch.as_tensor(combined_x[indices], dtype=torch.long)
            batch_lengths = torch.as_tensor(combined_seq_lengths[indices], dtype=torch.long)
            batch_labels = torch.as_tensor(combined_y_array[indices], dtype=torch.long)
            final_optimizer.zero_grad(set_to_none=True)
            logits = combined_model(batch_ids, batch_lengths)
            loss = final_loss(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            final_optimizer.step()

    training_seconds = time.perf_counter() - training_started
    # Test labels and features are first consumed after the final refit, and the
    # held-out test predictions/metrics are computed exactly once.
    test_texts = list(test_texts)
    test_y = _normalize_labels(test_labels, "test", n_classes)
    test_x, test_lengths = _encode([_tokenize(text) for text in test_texts], vocab, max_length)
    test_predictions = _predict(combined_model, test_x, test_lengths, config["batch_size"])
    test_metrics = classification_metrics(test_y, test_predictions, names)

    config["selected_epoch"] = best_epoch
    config["selected_validation_macro_f1"] = best_f1
    config["class_weights"] = [float(weight) for weight in class_weights]
    config["final_fit_epochs"] = best_epoch
    config["final_fit_data"] = "train+validation"
    config["vocabulary_size"] = len(vocab)
    config["seed"] = int(seed)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "neural_weights.npz"
    metadata_path = destination / "neural_metadata.json"
    # Store only numeric arrays: no executable/arbitrary Python objects are pickled.
    np.savez_compressed(
        weights_path,
        **{key: value.detach().cpu().numpy() for key, value in combined_model.state_dict().items()},
    )
    metadata = {
        "method": "scratch-trained word embedding + bidirectional GRU",
        "label_names": names,
        "label_to_id": {name: index for index, name in enumerate(names)},
        "vocabulary": vocab,
        "special_tokens": {
            "pad": _PAD_TOKEN,
            "pad_id": _PAD_ID,
            "unk": _UNK_TOKEN,
            "unk_id": _UNK_ID,
        },
        "tokenizer": "Unicode NFKC normalization, casefold, split on Unicode word characters",
        "selected_config": config,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "method": "scratch-trained word embedding + bidirectional GRU",
        "selected_config": config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "model_path": str(weights_path),
        "metadata_path": str(metadata_path),
        "training_seconds": float(training_seconds),
    }
