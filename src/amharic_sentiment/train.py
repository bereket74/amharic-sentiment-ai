from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import transformers

from amharic_sentiment.data import (
    LABEL_NAMES,
    download_dataset,
    load_and_audit_splits,
    save_audit,
)
from amharic_sentiment.metrics import bootstrap_macro_f1_ci
from amharic_sentiment.models import classical, neural, transformer

MODEL_MODULES = {
    "classical": classical,
    "neural": neural,
    "transformer": transformer,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_predictions(path: Path, y_true: list[int], y_pred: list[int], names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_row_id",
                "true_label_id",
                "true_label",
                "predicted_label_id",
                "predicted_label",
                "correct",
            ]
        )
        for index, (truth, prediction) in enumerate(zip(y_true, y_pred, strict=True)):
            writer.writerow(
                [index, truth, names[truth], prediction, names[prediction], truth == prediction]
            )


def _plot_model_comparison(results: dict[str, Any], path: Path, selected_name: str) -> None:
    names = list(results)
    scores = [results[name]["test_metrics"]["macro_f1"] for name in names]
    accuracies = [results[name]["test_metrics"]["accuracy"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, scores, width, label="Macro-F1", color="#1f5a85")
    bars2 = ax.bar(x + width / 2, accuracies, width, label="Accuracy", color="#5ba88a")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score")
    ax.set_xticks(x, [name.title() for name in names])
    ax.set_title("Amharic sentiment: fixed AfriSenti test split")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for tick, name in zip(ax.get_xticklabels(), names, strict=True):
        if name == selected_name:
            tick.set_fontweight("bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(metrics: dict[str, Any], path: Path, model_name: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5.3, 4.4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name.title()} — held-out test")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Number of posts")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_error_summary(best_name: str, metrics: dict[str, Any], path: Path) -> None:
    matrix = metrics["confusion_matrix"]
    errors: list[tuple[int, str, str]] = []
    for true_id, row in enumerate(matrix):
        for pred_id, count in enumerate(row):
            if true_id != pred_id and count:
                errors.append((count, LABEL_NAMES[true_id], LABEL_NAMES[pred_id]))
    errors.sort(reverse=True)
    lines = [
        f"# Aggregate error analysis: {best_name.title()}",
        "",
        (
            "This report deliberately omits tweet text. It summarizes the selected model's "
            "held-out errors without redistributing social-media content."
        ),
        "",
        "## Per-class recall",
        "",
        "| Class | Recall | Test examples |",
        "|---|---:|---:|",
    ]
    for label in LABEL_NAMES:
        row = metrics["per_class"][label]
        lines.append(f"| {label} | {row['recall']:.3f} | {row['support']} |")
    lines.extend(["", "## Most frequent confusions", ""])
    if errors:
        for count, true_name, pred_name in errors:
            lines.append(f"- **{true_name} → {pred_name}:** {count} posts")
    else:
        lines.append("- No misclassifications in the held-out test set.")
    lines.extend(
        [
            "",
            (
                "Counts are descriptive, not causal. Social-media sentiment includes sarcasm, "
                "mixed-language posts, cultural context, and annotator disagreement; row-level "
                "text is not included in this report."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    *,
    raw_dir: Path,
    run_dir: Path,
    reports_dir: Path,
    models: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    split_paths = download_dataset(raw_dir)
    splits, audit = load_and_audit_splits(split_paths)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_audit(audit, run_dir / "data_audit.json")
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_audit(audit, reports_dir / "data_audit.json")

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    output: dict[str, Any] = {}
    for model_name in models:
        print(f"\n=== Training {model_name} baseline ===", flush=True)
        result = MODEL_MODULES[model_name].run_experiment(
            train_texts=train["texts"],
            train_labels=train["labels"],
            val_texts=validation["texts"],
            val_labels=validation["labels"],
            test_texts=test["texts"],
            test_labels=test["labels"],
            output_dir=run_dir / model_name,
            label_names=LABEL_NAMES,
            seed=seed,
        )
        result["test_metrics"]["macro_f1_95pct_bootstrap_ci"] = bootstrap_macro_f1_ci(
            test["labels"], result["test_predictions"], len(LABEL_NAMES), seed=seed
        )
        _write_predictions(
            run_dir / model_name / "test_predictions.csv",
            test["labels"],
            result["test_predictions"],
            LABEL_NAMES,
        )
        output[model_name] = result
        print(
            f"{model_name}: validation macro-F1={result['val_metrics']['macro_f1']:.4f}; "
            f"test macro-F1={result['test_metrics']['macro_f1']:.4f}; "
            f"test accuracy={result['test_metrics']['accuracy']:.4f}; "
            f"elapsed={result['training_seconds']:.1f}s",
            flush=True,
        )

    # The best checkpoint is selected exclusively by validation macro-F1.
    selected_name = max(
        models,
        key=lambda name: (output[name]["val_metrics"]["macro_f1"], -models.index(name)),
    )
    selected = output[selected_name]
    metadata = {
        "project": "Amharic Sentiment AI",
        "dataset": audit,
        "label_names_and_ids": {name: index for index, name in enumerate(LABEL_NAMES)},
        "seed": seed,
        "selection_rule": (
            "highest validation macro-F1; fixed official test set evaluated once per candidate; "
            "test results did not choose the winner"
        ),
        "selected_model": selected_name,
        "selected_model_path": str(Path(selected["model_path"]).resolve()),
        "models": {
            name: {
                "method": result["method"],
                "selected_config": result["selected_config"],
                "validation_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
                "training_seconds": result["training_seconds"],
                "model_path": str(Path(result["model_path"]).resolve()),
            }
            for name, result in output.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "transformers": transformers.__version__,
        },
    }
    for destination in (run_dir / "benchmark_results.json", reports_dir / "benchmark_results.json"):
        destination.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
    (run_dir / "best_model.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_metric": "validation_macro_f1",
                "validation_macro_f1": selected["val_metrics"]["macro_f1"],
                "model_path": str(Path(selected["model_path"]).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_model_comparison(output, reports_dir / "model_comparison.png", selected_name)
    _plot_confusion_matrix(
        selected["test_metrics"], reports_dir / "best_confusion_matrix.png", selected_name
    )
    _write_error_summary(
        selected_name,
        selected["test_metrics"],
        reports_dir / "error_analysis.md",
    )
    summary_rows = []
    for name, result in output.items():
        metrics = result["test_metrics"]
        summary_rows.append(
            {
                "model": name,
                "method": result["method"],
                "validation_macro_f1": result["val_metrics"]["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1_ci_low": metrics["macro_f1_95pct_bootstrap_ci"][0],
                "test_macro_f1_ci_high": metrics["macro_f1_95pct_bootstrap_ci"][1],
                "training_seconds": result["training_seconds"],
                "selected_for_release": name == selected_name,
            }
        )
    with (reports_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSelected by validation macro-F1: {selected_name}", flush=True)
    print(f"Test metrics and plots: {reports_dir.resolve()}", flush=True)
    print(f"Model artifacts: {run_dir.resolve()}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Amharic sentiment baselines")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/afrisenti-amharic"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_MODULES), default=list(MODEL_MODULES)
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    run_dir = args.run_dir or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_benchmark(
        raw_dir=args.raw_dir,
        run_dir=run_dir,
        reports_dir=args.reports_dir,
        models=args.models,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
