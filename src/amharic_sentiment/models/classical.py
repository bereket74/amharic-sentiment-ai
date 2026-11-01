"""Classical word- and character-TF-IDF sentiment baseline for Amharic text."""

from __future__ import annotations

import os
import pickle
import time
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from amharic_sentiment.metrics import classification_metrics


def _make_pipeline(c_value: float, seed: int) -> Pipeline:
    """Build a deterministic sparse n-gram model without normalizing raw text."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=300_000,
                ),
            ),
        ],
        transformer_weights={"word_tfidf": 1.0, "char_tfidf": 1.0},
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    random_state=seed,
                    dual=True,
                    max_iter=5000,
                ),
            ),
        ]
    )


def run_experiment(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    output_dir: str,
    label_names: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Tune regularization on validation data, refit, and evaluate the test split."""
    started = time.perf_counter()
    c_grid = (0.25, 1.0, 4.0)
    validation_candidates: list[tuple[float, float, list[int]]] = []

    for c_value in c_grid:
        candidate = _make_pipeline(c_value, seed)
        candidate.fit(train_texts, train_labels)
        predictions = candidate.predict(val_texts).tolist()
        metrics = classification_metrics(val_labels, predictions, label_names)
        validation_candidates.append((metrics["macro_f1"], c_value, predictions))

    # In a validation tie, prefer the more regularized (smaller-C) model.
    best_macro_f1, selected_c, val_predictions = max(
        validation_candidates, key=lambda item: (item[0], -item[1])
    )
    val_metrics = classification_metrics(val_labels, val_predictions, label_names)

    combined_texts = list(train_texts) + list(val_texts)
    combined_labels = list(train_labels) + list(val_labels)
    final_model = _make_pipeline(selected_c, seed)
    final_model.fit(combined_texts, combined_labels)

    # The test labels are first used here, after tuning and the final refit.
    test_predictions = final_model.predict(test_texts).tolist()
    test_metrics = classification_metrics(test_labels, test_predictions, label_names)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    try:
        import joblib
    except ImportError:
        # The payload remains a standard pickle that joblib.load can read.
        with open(model_path, "wb") as model_file:
            pickle.dump(final_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(final_model, model_path)
    training_seconds = time.perf_counter() - started

    return {
        "method": "word_and_char_tfidf_linear_svc",
        "selected_config": {
            "classifier": "LinearSVC",
            "C": float(selected_c),
            "class_weight": "balanced",
            "word_ngram_range": [1, 2],
            "char_analyzer": "char_wb",
            "char_ngram_range": [3, 5],
            "validation_macro_f1": float(best_macro_f1),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "model_path": model_path,
        "training_seconds": float(training_seconds),
    }
