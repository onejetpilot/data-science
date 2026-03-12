from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DATA_FILE = "toxic_comments.csv"
REQUIRED_COLUMNS = {"text", "toxic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw toxic comments dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / DATA_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    toxic = pd.to_numeric(df["toxic"], errors="coerce")
    bad_target = int(toxic.isna().sum())
    if bad_target > 0:
        raise ValueError(f"Invalid target values in column 'toxic': {bad_target}")

    validated_path = args.output_dir / DATA_FILE
    shutil.copyfile(raw_path, validated_path)

    report = {
        "status": "ok",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_columns": missing_columns,
        "null_cells": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "class_balance": {str(int(k)): int(v) for k, v in toxic.astype(int).value_counts().to_dict().items()},
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[validate] Rows: {len(df)}")
    print(f"[validate] Saved file: {validated_path}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
