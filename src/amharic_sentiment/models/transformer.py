"""CPU fine-tuning experiment for Amharic sentiment with AfroXLMR-mini."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amharic_sentiment.metrics import classification_metrics

MODEL_NAME = "Davlan/afro-xlmr-mini"
MODEL_REVISION = "bc04038b969667884bd83cdd37bed9559c2c3d9c"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This experiment intentionally runs on CPU, where these settings are enough
    # to make the shuffled batches and parameter initialization reproducible.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _encode_texts(tokenizer: Any, texts: Sequence[str], max_length: int) -> TensorDataset:
    encoded = tokenizer(
        [str(text) for text in texts],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"])


def _predict(
    model: torch.nn.Module,
    dataset: TensorDataset,
    batch_size: int,
) -> list[int]:
    predictions: list[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for input_ids, attention_mask in loader:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return [int(value) for value in predictions]


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
    """Select an epoch on train/validation, refit on both, then score test once.

    Sentiment label IDs are kept exactly as supplied: positive=0, neutral=1,
    negative=2. ``label_names`` must therefore be ordered by those IDs.
    """
    n_classes = 3
    if len(label_names) != n_classes:
        raise ValueError("label_names must contain exactly three names ordered by label ID")
    if not train_texts or len(train_texts) != len(train_labels):
        raise ValueError("train_texts and train_labels must have the same nonzero length")
    if not val_texts or len(val_texts) != len(val_labels):
        raise ValueError("val_texts and val_labels must have the same nonzero length")

    def checked_labels(values: Sequence[int], split: str) -> list[int]:
        result = [int(value) for value in values]
        if any(value not in (0, 1, 2) for value in result):
            raise ValueError(f"{split} labels must be integer IDs 0, 1, or 2")
        return result

    train_y = checked_labels(train_labels, "train")
    val_y = checked_labels(val_labels, "validation")

    # Limit CPU thread contention on shared/oversubscribed machines.
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1) // 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits this setting before its inter-op pool is used.
        pass
    _seed_everything(seed)

    # Train-only profile: the pinned training split's 95th/99th token lengths are
    # 63/70, so 72 covers over 99% without paying for 96 padded positions.
    max_length = 72
    batch_size = 8
    max_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.01
    patience = 1
    id2label = {index: str(name) for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

    destination = Path(output_dir)
    model_dir = destination / "model"
    destination.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Preserve XLM-R's WhitespaceSplit/Metaspace tokenizer; the Mistral regex fix is not applicable.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, fix_mistral_regex=False
    )
    train_tokens = _encode_texts(tokenizer, train_texts, max_length)
    val_tokens = _encode_texts(tokenizer, val_texts, max_length)

    # Stage 1: train only on the initial training subset and select an epoch
    # using validation macro-F1. Preserve that stage's metrics and predictions.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_dataset = TensorDataset(
        train_tokens.tensors[0],
        train_tokens.tensors[1],
        torch.tensor(train_y, dtype=torch.long),
    )
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, Any] | None = None
    best_val_predictions: list[int] | None = None
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_predictions = _predict(model, val_tokens, batch_size)
        val_metrics = classification_metrics(val_y, val_predictions, list(label_names))
        score = float(val_metrics["macro_f1"])
        if score > best_macro_f1:
            best_macro_f1 = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_val_metrics = val_metrics
            best_val_predictions = val_predictions
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_val_metrics is None or best_val_predictions is None:
        raise RuntimeError("Training completed without producing a validation checkpoint")

    # Retain the historic explicit checkpoint, but use the new refit below as
    # the deployable Hugging Face model bundle.
    best_state_path = destination / "best_state.pt"
    torch.save(best_state, best_state_path)

    # Stage 2: discard the selection model and initialize a fresh classifier
    # from the same pretrained base. Refit on train+validation for exactly the
    # selected number of epochs, with no validation or early stopping.
    del model, optimizer, train_loader, train_dataset
    _seed_everything(seed)
    refit_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )
    refit_model.to(torch.device("cpu"))
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    combined_dataset = TensorDataset(
        torch.cat((train_tokens.tensors[0], val_tokens.tensors[0]), dim=0),
        torch.cat((train_tokens.tensors[1], val_tokens.tensors[1]), dim=0),
        torch.tensor(train_y + val_y, dtype=torch.long),
    )
    refit_generator = torch.Generator().manual_seed(seed)
    refit_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=refit_generator,
    )
    for _ in range(best_epoch):
        refit_model.train()
        for input_ids, attention_mask, labels in refit_loader:
            refit_optimizer.zero_grad(set_to_none=True)
            outputs = refit_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(refit_model.parameters(), max_norm=1.0)
            refit_optimizer.step()

    training_seconds = float(time.perf_counter() - started)

    # Persist the final train+validation refit as the deployable model/tokenizer.
    refit_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    selected_config: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "selected_epoch": best_epoch,
        "early_stopping_patience": patience,
        "best_validation_macro_f1": best_macro_f1,
        "selection_protocol": "train subset; select best epoch by validation macro-F1",
        "final_refit_protocol": {
            "initialization": "fresh sequence-classification model from the same pretrained base",
            "training_data": "concatenated train and validation subsets",
            "epochs": best_epoch,
            "epoch_selection_or_early_stopping": False,
            "deployable_artifact": str(model_dir),
        },
        "seed": int(seed),
        "device": "cpu",
        "label_names": [str(name) for name in label_names],
        "label2id": label2id,
        "id2label": {str(index): name for index, name in id2label.items()},
        "best_state_path": str(best_state_path),
    }
    with (destination / "training_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(selected_config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")

    # Do not inspect or encode any test data until the final refit is complete.
    if not test_texts or len(test_texts) != len(test_labels):
        raise ValueError("test_texts and test_labels must have the same nonzero length")
    test_y = checked_labels(test_labels, "test")
    test_tokens = _encode_texts(tokenizer, test_texts, max_length)
    test_predictions = _predict(refit_model, test_tokens, batch_size)
    test_metrics = classification_metrics(test_y, test_predictions, list(label_names))

    return {
        "method": "fine-tuned-transformer",
        "selected_config": selected_config,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": best_val_predictions,
        "test_predictions": test_predictions,
        "model_path": str(model_dir),
        "training_seconds": training_seconds,
    }
