from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_FILE = "taxi.csv"
TEST_SIZE = 0.1
IMPORTANT_LAGS = [1, 2, 3, 24, 168]
ROLLING_WINDOWS = [24, 168]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed train/test splits for taxi time-series.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.validated_dir / DATA_FILE, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Ресемплирование по часу, как в ноутбуке.
    df = df.resample("1h").sum()

    # Лаговые признаки.
    for lag in IMPORTANT_LAGS:
        df[f"lag_{lag}"] = df["num_orders"].shift(lag)

    # Скользящие средние.
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df["num_orders"].shift(1).rolling(window=window).mean()

    split_idx = int(len(df) * (1 - TEST_SIZE))
    train = df.iloc[:split_idx].dropna().copy()
    test = df.iloc[split_idx:].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    full_path = args.output_dir / "dataset.parquet"
    try:
        train.to_parquet(train_path)
        test.to_parquet(test_path)
        df.to_parquet(full_path)
    except Exception:
        train_path = args.output_dir / "train.csv"
        test_path = args.output_dir / "test.csv"
        full_path = args.output_dir / "dataset.csv"
        train.to_csv(train_path)
        test.to_csv(test_path)
        df.to_csv(full_path)

    manifest = {
        "dataset_path": str(full_path.resolve()),
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target": "num_orders",
        "lags": IMPORTANT_LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "test_size": TEST_SIZE,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved test: {test_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Rows train/test: {len(train)}/{len(test)}")


if __name__ == "__main__":
    main()
