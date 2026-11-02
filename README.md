# Amharic Sentiment AI

A reproducible three-way sentiment benchmark for Amharic social-media text. This project compares a classical TF-IDF classifier, a compact neural network trained from scratch, and a fine-tuned African-language transformer. It uses the official AfriSenti Amharic (`amh`) split and deliberately does not redistribute tweet text.

## Current benchmark

The final model is chosen by **validation macro-F1**, not by test performance. In the completed run, AfroXLM-R-mini scored highest on validation (0.536) and is the selected release candidate. Each candidate was refit on train+validation before the official test evaluation.

An initial CPU attempt was interrupted before transformer evaluation, but it had already scored the classical and neural candidates. They were rerun with the same seed and unchanged settings to create a complete report. No settings were changed based on those test scores; nevertheless, those two test outcomes were observed twice, so the public test should be treated as a fixed benchmark rather than a fully untouched blind set for those candidates.

| Approach | Validation macro-F1 | Test macro-F1 | Test accuracy | Test macro-F1 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Classical TF-IDF + LinearSVC | 0.516 | 0.429 | 0.441 | 0.408–0.449 |
| Scratch-trained BiGRU | 0.488 | 0.421 | 0.454 | 0.400–0.444 |
| AfroXLM-R-mini transformer | **0.536** | **0.466** | 0.460 | 0.444–0.487 |

The selected transformer’s test F1 is 0.573 for positive, 0.279 for neutral, and 0.544 for negative. Neutral recall is high (0.839), but its precision is only 0.167; 751 negative posts were predicted as neutral. This is an important failure mode, not a claim of general-purpose sentiment accuracy. Classical and neural test outcomes were observed in an interrupted preliminary run and again in the completed run; they were not used for tuning, but are not fully untouched blind evaluations. The transformer took 6,694 CPU seconds (about 1 h 52 min); the classical and neural runs took about 15 and 174 seconds.

See [the test comparison plot](reports/model_comparison.png), [confusion matrix](reports/best_confusion_matrix.png), and aggregate [error analysis](reports/error_analysis.md). The model card and Hugging Face Space bundle are assembled with `scripts/build_release.py`.

## What is compared

- **Classical:** word unigrams/bigrams and character-boundary n-grams with TF-IDF, followed by class-balanced LinearSVC. Regularization is selected on validation macro-F1.
- **Neural:** a word-vocabulary embedding plus bidirectional GRU trained from scratch. Early stopping selects a fixed epoch count using validation macro-F1.
- **Transformer:** `Davlan/afro-xlmr-mini`, fine-tuned as a three-class classifier. The upstream model revision is pinned in `src/amharic_sentiment/models/transformer.py`; validation selects a fixed epoch count. To fit CPU limits, the run caps input at 72 tokens (batch 8; at most 3 epochs; patience 1). A train-only tokenizer profile gave p95/p99 lengths of 63/70 tokens; validation and test lengths were not used to choose the cap. Reproduce the profile with `scripts/profile_train_token_lengths.py`.

After selection, each final model is freshly refit on the combined train and validation sets for its selected configuration/epoch count. No test data is used for vocabulary, tuning, early stopping, or refitting. The winner is selected only by validation macro-F1. Test metrics are reported transparently for all candidates and are not used to select the released model.

## Data and limits

The dataset is the official SemEval-2023 AfriSenti Amharic subset, labeled positive, neutral, or negative. Its source repository states **CC BY 4.0**. Files are downloaded from the immutable commit pinned in `src/amharic_sentiment/data.py`; raw text stays under the ignored `data/raw/` directory. The source SHA-256 values, split counts, label counts, and duplicate audit are saved in `reports/data_audit.json`.

Exact normalized duplicates are audited across train, validation, and test. Duplicate rows are removed from train/validation where needed, but the official test rows remain unchanged. Metrics include macro-F1, weighted-F1, balanced accuracy, per-class scores, a confusion matrix, and a deterministic percentile-bootstrap interval for test macro-F1.

The split label distributions shift sharply: negative posts are 25.86% of train and 25.92% of validation, but 66.88% of test; neutral posts fall from about 51.9% to 11.21%. Treat held-out metrics as results under this official distribution shift, not as an i.i.d. estimate of deployment performance. Accuracy alone can hide weak performance on minority classes. Social-media language includes sarcasm, regional variation, spelling variation, code-switching, and context-dependent meaning. The model output is neither calibrated confidence nor ground truth about a person. Do not use it for high-impact decisions or profiling.

See [data and model provenance](docs/sources.md) and the [architecture diagram](docs/architecture.png).

## Install and reproduce

Python 3.12 is tested. The checked environment uses CPU PyTorch 2.15.0.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
pytest -q
```

A full dependency snapshot for the tested environment is in `requirements.lock`. The project downloads the three TSVs from the pinned upstream revision if they are not already present and records their hashes in the audit report. The token-length profiling script reads only `train.tsv` and never prints post contents. To train all candidates with seed 42:

```bash
python -m amharic_sentiment.train --run-dir runs/benchmark --reports-dir reports --seed 42
```

A run writes each model and test predictions under `runs/`; the per-row file includes only row IDs and labels, never tweet text. `runs/` and raw data are ignored by Git. Aggregate metrics, figures, and error counts in `reports/` contain no source posts. The completed local run for this release is `runs/benchmark-final-cpu72-20261101`.

## Run the local demo

After training, use the selected model path printed by the training script or listed in `runs/benchmark/benchmark_results.json`:

```bash
python app.py --model-type transformer --model-path runs/benchmark-final-cpu72-20261101/transformer/model
```

Replace `transformer` and its path with `classical` plus the `model.joblib` path, or `neural` plus the `neural_weights.npz` path, if validation selects a different model. The Gradio demo runs locally at `http://127.0.0.1:7860` and does not create a public share link.

The public model is [klohdfhk/amharic-sentiment-ai](https://huggingface.co/klohdfhk/amharic-sentiment-ai). The interactive [Hugging Face Static Space](https://huggingface.co/spaces/klohdfhk/amharic-sentiment-demo) runs a quantized ONNX model in the visitor's browser; it does not use an inference server or paid hardware. Its first use downloads about 141 MB of model and tokenizer files. The GitHub source is [bereket74/amharic-sentiment-ai](https://github.com/bereket74/amharic-sentiment-ai).

### Rebuild the browser release

The optional ONNX tools are separate from the core training environment. These commands export the selected transformer, evaluate the quantized variant on the official splits (writing aggregate metrics only), then assemble and publish the model and static Space:

```bash
pip install -r requirements.browser.txt
python scripts/export_browser_model.py --run-dir runs/benchmark-final-cpu72-20261101 --output-dir release/browser-model-2026-11-02
python scripts/evaluate_browser_model.py --run-dir runs/benchmark-final-cpu72-20261101 --raw-dir data/raw/afrisenti-amharic --quantized-model-dir release/browser-model-2026-11-02/onnx-int8-per-channel --output release/browser_model_evaluation.json
python scripts/build_release.py --run-dir runs/benchmark-final-cpu72-20261101 --onnx-model release/browser-model-2026-11-02/onnx-int8-per-channel/model_quantized.onnx --browser-evaluation release/browser_model_evaluation.json --output-dir release/static-2026-11-02
HF_TOKEN="$HF_TOKEN" python scripts/publish_huggingface.py --release-dir release/static-2026-11-02
```

The original Gradio/Docker Space attempt returned HTTP 402 because this account could not use its hosted CPU runtime. No paid plan or hardware was enabled; the free Static Space avoids that limitation by running inference locally. The 8-bit variant scored 0.659 validation macro-F1 and 0.456 test macro-F1 versus 0.666 and 0.466 for the full transformer, with 94.5% agreement on test labels. The official test split is distribution-shifted, and predictions are not calibrated probabilities.

The GitHub repository contains source, tests, documentation, and aggregate evaluation only; raw social-media posts and model weights are excluded. The `.env` file is ignored by Git; never commit a real token. The Hugging Face publisher reads `HF_TOKEN` from the environment and never stores or prints it.

## License

The MIT `LICENSE` applies to original project code only. The AfriSenti data remain under their upstream CC BY 4.0 terms, and the AfroXLM-R base model retains its own stated license. Any published fine-tuned model card must preserve dataset attribution and describe the model's limits. No raw tweet text is included in releases.
