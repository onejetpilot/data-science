from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = ["distance", "insurance_premium", "vehicle_age"]
CATEGORICAL_COLUMNS = [
    "weather_1",
    "road_surface",
    "lighting",
    "county_city_location",
    "county_location",
    "direction",
    "location_type",
    "road_condition_1",
    "vehicle_type",
    "vehicle_transmission",
]
BINARY_COLUMNS = ["cellphone_in_use"]
TARGET_COLUMN = "at_fault"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed train/val dataset for collisions.")
    parser.add_argument(
        "--validated-csv",
        type=Path,
        default=Path("data/validated/df_dtp_validated.csv"),
        help="Path to validated df_dtp csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_table(df: pd.DataFrame, base_path: Path) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]

    # Удаляем технический индекс, если пришел из ноутбука.
    if "unnamed: 0" in data.columns:
        data = data.drop(columns=["unnamed: 0"])

    # Приводим типы числовых и бинарных признаков.
    for col in NUMERIC_COLUMNS + BINARY_COLUMNS + [TARGET_COLUMN]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Базовая очистка: target должен быть бинарным, числовые пропуски заполняем медианой.
    data = data[data[TARGET_COLUMN].isin([0, 1])]
    for col in NUMERIC_COLUMNS:
        data[col] = data[col].fillna(data[col].median())
    for col in BINARY_COLUMNS:
        data[col] = data[col].fillna(0).astype(int)

    # Категориальные признаки заполняем и приводим к string.
    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna("unknown").astype(str)

    model_columns = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    data = data[model_columns].dropna(subset=[TARGET_COLUMN])
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return data


def main() -> None:
    # 1) Загружаем validated слой и выполняем очистку.
    args = parse_args()
    if not args.validated_csv.exists():
        raise FileNotFoundError(f"Validated csv not found: {args.validated_csv}")
    raw = pd.read_csv(args.validated_csv)
    dataset = clean_dataframe(raw)

    # 2) Делим на train/val со стратификацией по целевому классу.
    train_df, val_df = train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset[TARGET_COLUMN],
    )

    # 3) Сохраняем выборки и manifest признаков.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = write_table(train_df, args.output_dir / "train")
    val_path = write_table(val_df, args.output_dir / "val")

    manifest = {
        "target": TARGET_COLUMN,
        "numeric_features": NUMERIC_COLUMNS,
        "binary_features": BINARY_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved val: {val_path}")
    print(f"[build_dataset] Saved feature manifest: {manifest_path}")
    print(f"[build_dataset] Rows train={len(train_df)} val={len(val_df)}")


if __name__ == "__main__":
    main()
