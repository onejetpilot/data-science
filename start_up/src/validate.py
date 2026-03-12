from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


TRAIN_FILE = "kaggle_startups_train_28062024.csv"
TEST_FILE = "kaggle_startups_test_28062024.csv"
CITIES_FILE = "worldcitiespop.csv"

TRAIN_REQUIRED = {
    "name",
    "category_list",
    "funding_total_usd",
    "funding_rounds",
    "founded_at",
    "first_funding_at",
    "last_funding_at",
    "country_code",
    "state_code",
    "region",
    "city",
    "status",
}

TEST_REQUIRED = {
    "name",
    "category_list",
    "funding_total_usd",
    "funding_rounds",
    "founded_at",
    "first_funding_at",
    "last_funding_at",
    "country_code",
    "state_code",
    "region",
    "city",
}

CITIES_REQUIRED = {"City", "Population"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw startup datasets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"))
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> list[str]:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")
    return missing


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.raw_dir / TRAIN_FILE
    test_path = args.raw_dir / TEST_FILE
    cities_path = args.raw_dir / CITIES_FILE
    for file_path in [train_path, test_path, cities_path]:
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    cities_df = pd.read_csv(cities_path, low_memory=False)

    validate_columns(train_df, TRAIN_REQUIRED, "train")
    validate_columns(test_df, TEST_REQUIRED, "test")
    validate_columns(cities_df, CITIES_REQUIRED, "cities")

    shutil.copyfile(train_path, args.output_dir / TRAIN_FILE)
    shutil.copyfile(test_path, args.output_dir / TEST_FILE)
    shutil.copyfile(cities_path, args.output_dir / CITIES_FILE)

    report = {
        "status": "ok",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cities_rows": int(len(cities_df)),
        "train_columns": int(len(train_df.columns)),
        "test_columns": int(len(test_df.columns)),
        "cities_columns": int(len(cities_df.columns)),
        "train_null_cells": int(train_df.isna().sum().sum()),
        "test_null_cells": int(test_df.isna().sum().sum()),
        "cities_null_cells": int(cities_df.isna().sum().sum()),
        "train_duplicates": int(train_df.duplicated().sum()),
        "test_duplicates": int(test_df.duplicated().sum()),
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[validate] Train rows: {len(train_df)}")
    print(f"[validate] Test rows: {len(test_df)}")
    print(f"[validate] Cities rows: {len(cities_df)}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
