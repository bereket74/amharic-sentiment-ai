from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO_ID = "klohdfhk/amharic-sentiment-ai"


def _metric_rows(metrics: dict[str, Any]) -> str:
    return "\n".join(
        f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | "
        f"{item['f1']:.3f} | {item['support']} |"
        for label, item in metrics["per_class"].items()
    )


def _public_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove local-only filesystem paths from the public release metadata."""
    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"best_state_path", "model_path", "metadata_path"}:
            continue
        if key == "deployable_artifact":
            cleaned[key] = "model repository root"
        elif isinstance(value, dict):
            cleaned[key] = _public_training_config(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _public_training_config(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _inference_example(model_type: str) -> list[str]:
    return [
        "## Example inference",
        "",
        "With Python 3.12, install the pinned dependencies, then download this model repo:",
        "",
        "```bash",
        "pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu",
        "```",
        "",
        "```python",
        "from pathlib import Path",
        "import sys",
        "from huggingface_hub import snapshot_download",
        f'repo_id = "{MODEL_REPO_ID}"',
        f'model_type = "{model_type}"',
        "model_dir = Path(snapshot_download(repo_id=repo_id))",
        "sys.path.insert(0, str(model_dir / 'src'))",
        "from amharic_sentiment.inference import SentimentPredictor",
        "if model_type == 'transformer':",
        "    artifact = model_dir",
        "else:",
        "    filename = 'model.joblib' if model_type == 'classical' else 'neural_weights.npz'",
        "    artifact = model_dir / filename",
        "predictor = SentimentPredictor(model_type, artifact)",
        'print(predictor.predict_label("ይህ በጣም ጥሩ ነው"))',
        "```",
        "",
        "## Demo availability",
        "",
        (
            "No hosted Gradio demo is currently available. The inference code "
            "is included in this repository."
        ),
        "",
    ]


def _model_card(metadata: dict[str, Any], model_type: str) -> str:
    selected = metadata["models"][model_type]
    metrics = selected["test_metrics"]
    low, high = metrics["macro_f1_95pct_bootstrap_ci"]
    config = selected["selected_config"]
    lines = [
        "---",
        "license: cc-by-4.0",
        "language:",
        "  - am",
        "pipeline_tag: text-classification",
        "tags:",
        "  - amharic",
        "  - sentiment-analysis",
        "  - afrisenti",
        "---",
        "",
        "# Amharic Sentiment AI",
        "",
        "A three-class Amharic sentiment classifier trained on the official AfriSenti",
        "Amharic benchmark. The release contains model files, code for inference, and",
        "reproducibility metadata; it does not contain source tweets.",
        "",
        "## Labels",
        "",
        "- `positive` (ID 0)",
        "- `neutral` (ID 1)",
        "- `negative` (ID 2)",
        "",
        "## Training and selection",
        "",
        f"- Selected approach: **{selected['method']}** (`{model_type}`)",
        "- Selection: highest validation macro-F1 across candidates; test results did not",
        "  choose the released model.",
        "- Final fit: fresh refit on train+validation using settings or epoch count selected",
        "  on validation; no test use during selection or refitting.",
        f"- Seed: `{metadata['seed']}`",
    ]
    if model_type == "transformer":
        lines.extend(
            [
                f"- Base model: [`{config['model_name']}`]",
                f"  (https://huggingface.co/{config['model_name']})",
                f"- Base revision: `{config['model_revision']}`",
            ]
        )
    lines.extend(
        [
            (
                "- Dataset: [AfriSenti Amharic](https://github.com/afrisenti-semeval/"
                "afrisent-semeval-2023), pinned commit "
                f"`{metadata['dataset']['source_commit']}`; license CC BY 4.0."
            ),
            "- Source SHA-256 and split-overlap audit: [`data_audit.json`](./data_audit.json).",
            "",
            "## Held-out test results",
            "",
            (
                f"The official test split contains "
                f"{metadata['dataset']['final_split_rows']['test']:,} posts. Macro-F1 is "
                "emphasized because class frequencies are imbalanced."
            ),
            "",
            "| Metric | Score |",
            "|---|---:|",
            f"| Accuracy | {metrics['accuracy']:.3f} |",
            f"| Macro-F1 | {metrics['macro_f1']:.3f} |",
            f"| Weighted-F1 | {metrics['weighted_f1']:.3f} |",
            f"| Balanced accuracy | {metrics['balanced_accuracy']:.3f} |",
            f"| Macro-F1 95% bootstrap interval | {low:.3f}–{high:.3f} |",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
            _metric_rows(metrics),
            "",
            (
                "The test labels are distribution-shifted: 66.9% negative and 11.2% neutral, "
                "versus about 25.9% negative and 51.9% neutral in train/validation."
            ),
            (
                "Classical and neural test scores were also observed in an earlier interrupted "
                "run with identical settings; no hyperparameters were changed from those scores."
            ),
            "The bootstrap interval reflects test-row resampling for this fixed model, not",
            "variation across training seeds.",
            "",
            *_inference_example(model_type),
            "## Intended use and limitations",
            "",
            "This model is for research and educational demonstrations of Amharic sentiment",
            "classification. Social-media sentiment may not generalize across domains or time.",
            "Sarcasm, dialects, spelling variation, code-switching, cultural context, and",
            "annotation disagreement can affect results. Predictions are not calibrated",
            "probabilities and must not be used for high-impact decisions or profiling.",
            "",
            "## Data, attribution, and licensing",
            "",
            "AfriSenti was created for SemEval-2023 Task 12; the upstream repository states",
            "CC BY 4.0. Please cite and attribute the dataset authors; see [Muhammad et al.,",
            "EMNLP 2023](https://aclanthology.org/2023.emnlp-main.862/). This release does not",
            "bundle tweets. The base encoder's license and attribution remain applicable.",
            "",
            "Original project code is MIT licensed. Fine-tuned weights are released under",
            "CC BY 4.0 with attribution. This is a project release choice, not legal advice.",
            "",
            "## Reproducibility",
            "",
            "This card is generated from the benchmark results and the accompanying training",
            "config and audit. It reports one fixed-seed run; it does not claim multi-seed",
            "statistical significance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_model_artifact(model_type: str, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if model_type == "transformer":
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return
    if model_type == "classical":
        shutil.copy2(source, destination / "model.joblib")
        return
    shutil.copy2(source, destination / "neural_weights.npz")
    metadata = source.with_name("neural_metadata.json")
    shutil.copy2(metadata, destination / metadata.name)


def build(run_dir: Path, release_dir: Path) -> None:
    results_path = run_dir / "benchmark_results.json"
    metadata = json.loads(results_path.read_text(encoding="utf-8"))
    model_type = metadata["selected_model"]
    source = Path(metadata["models"][model_type]["model_path"])
    if release_dir.exists():
        raise FileExistsError(f"Release output already exists: {release_dir}; choose a new path")
    model_dir = release_dir / "model"
    _copy_model_artifact(model_type, source, model_dir)
    source_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(ROOT / "src", model_dir / "src", ignore=source_ignore)
    (model_dir / "README.md").write_text(_model_card(metadata, model_type), encoding="utf-8")
    shutil.copy2(ROOT / "requirements.lock", model_dir / "requirements.lock")
    (model_dir / "data_audit.json").write_text(
        json.dumps(metadata["dataset"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected = metadata["models"][model_type]
    (model_dir / "training_config.json").write_text(
        json.dumps(
            _public_training_config(selected["selected_config"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "validation_metrics": selected["validation_metrics"],
                "test_metrics": selected["test_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    space_dir = release_dir / "space"
    space_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "space" / "app.py", space_dir / "app.py")
    shutil.copy2(ROOT / "space" / "requirements.txt", space_dir / "requirements.txt")
    space_readme = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    (space_dir / "README.md").write_text(space_readme, encoding="utf-8")
    (space_dir / "model_config.json").write_text(
        json.dumps({"model_type": model_type}, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copytree(ROOT / "src", space_dir / "src", ignore=source_ignore)
    print(f"Model type: {model_type}")
    print(f"Model bundle: {model_dir.resolve()}")
    print(f"Space bundle: {space_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the selected model and demo for Hugging Face"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release" / "2026-11-01")
    args = parser.parse_args()
    build(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
