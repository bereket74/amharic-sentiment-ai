from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def _publish_folder(
    api: HfApi,
    source: Path,
    repo_id: str,
    repo_type: str,
    *,
    space_sdk: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Any:
    if not source.is_dir():
        raise FileNotFoundError(f"Release folder does not exist: {source}")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True,
        **({"space_sdk": space_sdk} if space_sdk else {}),
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(source),
        commit_message="Publish Amharic Sentiment AI release",
        **({"allow_patterns": allow_patterns} if allow_patterns else {}),
    )
    return commit


def _assert_public_or_missing(api: HfApi, repo_id: str, repo_type: str) -> bool:
    try:
        info = api.model_info(repo_id) if repo_type == "model" else api.space_info(repo_id)
    except RepositoryNotFoundError:
        return False
    if info.private:
        raise RuntimeError(
            f"Refusing to upload into the existing private {repo_type} repository {repo_id}."
        )
    return True


def publish(release_dir: Path, model_id: str, space_id: str, token: str) -> None:
    if not token:
        raise ValueError("Set HF_TOKEN to a project-specific Hugging Face write token.")
    api = HfApi(token=token)
    model_exists = _assert_public_or_missing(api, model_id, "model")
    _assert_public_or_missing(api, space_id, "space")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk="static",
    )
    model_patterns = None
    if model_exists:
        model_patterns = [
            "README.md",
            "browser_model_evaluation.json",
            "onnx/model_quantized.onnx",
            "src/amharic_sentiment/inference.py",
        ]
        print(
            "Updating existing model repo with the new ONNX artifact, evaluation, "
            "model card, and inference helper only.",
            flush=True,
        )
    model_commit = _publish_folder(
        api,
        release_dir / "model",
        model_id,
        "model",
        allow_patterns=model_patterns,
    )
    space_commit = _publish_folder(
        api,
        release_dir / "space",
        space_id,
        "space",
        space_sdk="static",
    )
    model_info = api.model_info(model_id)
    space_info = api.space_info(space_id)
    if model_info.private or space_info.private:
        raise RuntimeError("The uploaded repository or Space is private, not publicly published.")
    print(f"Model: https://huggingface.co/{model_id} (commit {model_commit.oid})")
    print(f"Static Space: https://huggingface.co/spaces/{space_id} (commit {space_commit.oid})")
    print("Space runtime: static HTML; no paid hardware requested")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the prepared Amharic sentiment model and Space to Hugging Face"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="klohdfhk/amharic-sentiment-ai")
    parser.add_argument("--space-id", default="klohdfhk/amharic-sentiment-demo")
    args = parser.parse_args()
    publish(args.release_dir, args.model_id, args.space_id, os.environ.get("HF_TOKEN", ""))


if __name__ == "__main__":
    main()
