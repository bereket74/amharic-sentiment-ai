"""Inference helpers for released Amharic sentiment models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amharic_sentiment.models.neural import (
    _BiGRUClassifier,
    _encode,
    _tokenize,
)

LABEL_NAMES = ["positive", "neutral", "negative"]


def _training_max_length(model_path: Path | str, source: str) -> int | None:
    if isinstance(model_path, Path):
        candidates = (
            model_path / "training_config.json",
            model_path.parent / "training_config.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                value = json.loads(candidate.read_text(encoding="utf-8")).get("max_length")
                return int(value) if value is not None else None
        return None
    try:
        config_path = Path(hf_hub_download(repo_id=source, filename="training_config.json"))
    except EntryNotFoundError:
        return None
    value = json.loads(config_path.read_text(encoding="utf-8")).get("max_length")
    return int(value) if value is not None else None


class SentimentPredictor:
    """Load one trained model once and reuse it for low-latency single predictions."""

    def __init__(
        self,
        model_type: str,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if model_type not in {"classical", "neural", "transformer"}:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model_type = model_type
        self.device = torch.device(device)
        self.model_path = (
            Path(model_path)
            if model_type != "transformer" or Path(model_path).exists()
            else str(model_path)
        )
        self.model: Any
        self.tokenizer: Any = None
        self.metadata: dict[str, Any] = {}
        if model_type == "classical":
            self.model = joblib.load(self.model_path)
        elif model_type == "transformer":
            source = str(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(source)
            self.model.to(self.device)
            self.model.eval()
            self.metadata["max_length"] = _training_max_length(self.model_path, source) or int(
                getattr(self.model.config, "max_position_embeddings", 128)
            )
        else:
            weights_path = Path(model_path)
            metadata_path = weights_path.with_name("neural_metadata.json")
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            config = self.metadata["selected_config"]
            self.vocab = {
                str(token): int(index) for token, index in self.metadata["vocabulary"].items()
            }
            self.max_length = int(config["sequence_length"])
            self.model = _BiGRUClassifier(
                vocab_size=len(self.vocab),
                embedding_dim=int(config["embedding_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                dropout=float(config["dropout"]),
                n_classes=len(self.metadata["label_names"]),
            )
            with np.load(weights_path, allow_pickle=False) as saved:
                state = {
                name: torch.tensor(np.array(saved[name], copy=True)) for name in saved.files
            }
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device)
            self.model.eval()

    def predict_label(self, text: str) -> str:
        """Return only the predicted sentiment label; confidence is not calibrated."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a non-empty text sample.")
        if len(text) > 5000:
            raise ValueError("Text is too long; enter at most 5,000 characters.")
        if self.model_type == "classical":
            label_id = int(self.model.predict([text])[0])
        elif self.model_type == "neural":
            encoded, lengths = _encode([_tokenize(text)], self.vocab, self.max_length)
            with torch.inference_mode():
                logits = self.model(
                    torch.as_tensor(encoded, dtype=torch.long, device=self.device),
                    torch.as_tensor(lengths, dtype=torch.long, device=self.device),
                )
            label_id = int(logits.argmax(dim=1).item())
        else:
            max_length = min(int(self.metadata["max_length"]), 512)
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            label_id = int(logits.argmax(dim=-1).item())
        if label_id < 0 or label_id >= len(LABEL_NAMES):
            raise RuntimeError(f"Model returned invalid label ID {label_id}.")
        return LABEL_NAMES[label_id]
