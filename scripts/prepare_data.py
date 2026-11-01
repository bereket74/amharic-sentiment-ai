from __future__ import annotations

import argparse
import json
from pathlib import Path

from amharic_sentiment.data import download_dataset, load_and_audit_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and audit Amharic AfriSenti data")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/afrisenti-amharic"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    paths = download_dataset(args.raw_dir)
    _, audit = load_and_audit_splits(paths)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
