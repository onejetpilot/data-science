from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FILE = "toxic_comments.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed train/test splits for toxic comments.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def normalize_text(text: object) -> str:
    value = str(text).lower()
    value = re.sub(r"[^a-z\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.validated_dir / DATA_FILE)
    df["text"] = df["text"].astype(str)
    df["toxic"] = pd.to_numeric(df["toxic"], errors="coerce")
    df = df.dropna(subset=["toxic"]).copy()
    df["toxic"] = df["toxic"].astype(int)
    df["clean_text"] = df["text"].apply(normalize_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    x = df[["clean_text"]]
    y = df["toxic"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_df = x_train.copy()
    train_df["toxic"] = y_train.values
    test_df = x_test.copy()
    test_df["toxic"] = y_test.values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    full_path = args.output_dir / "dataset.parquet"
    try:
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        df[["clean_text", "toxic"]].to_parquet(full_path, index=False)
    except Exception:
        train_path = args.output_dir / "train.csv"
        test_path = args.output_dir / "test.csv"
        full_path = args.output_dir / "dataset.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        df[["clean_text", "toxic"]].to_csv(full_path, index=False)

    manifest = {
        "dataset_path": str(full_path.resolve()),
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "target": "toxic",
        "text_column": "clean_text",
        "test_size": TEST_SIZE,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved test: {test_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
