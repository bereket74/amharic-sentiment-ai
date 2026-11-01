import json

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from amharic_sentiment.inference import LABEL_NAMES, SentimentPredictor, _training_max_length
from amharic_sentiment.models.neural import _BiGRUClassifier


def test_transformer_inference_reads_saved_training_max_length(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "training_config.json").write_text(json.dumps({"max_length": 72}), encoding="utf-8")

    assert _training_max_length(model_dir, str(model_dir)) == 72


def test_classical_joblib_artifact_loads_for_prediction(tmp_path):
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", DummyClassifier(strategy="constant", constant=1)),
        ]
    )
    pipeline.fit(["positive word", "neutral word", "negative word"], [0, 1, 2])
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    predictor = SentimentPredictor("classical", model_path)

    assert predictor.predict_label("በጣም ጥሩ") == "neutral"


def test_neural_numeric_checkpoint_loads_for_prediction(tmp_path):
    torch.manual_seed(13)
    config = {
        "embedding_dim": 8,
        "hidden_dim": 4,
        "dropout": 0.0,
        "sequence_length": 6,
    }
    vocabulary = {"<pad>": 0, "<unk>": 1, "እጅግ": 2, "ጥሩ": 3}
    model = _BiGRUClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        n_classes=3,
    )
    weights_path = tmp_path / "neural_weights.npz"
    np.savez_compressed(
        weights_path,
        **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
    )
    metadata = {
        "label_names": LABEL_NAMES,
        "vocabulary": vocabulary,
        "selected_config": config,
    }
    weights_path.with_name("neural_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    predictor = SentimentPredictor("neural", weights_path)

    assert predictor.predict_label("እጅግ ጥሩ") in LABEL_NAMES
