from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = ["registration_year", "power", "kilometer", "registration_month", "car_age"]
CATEGORICAL_COLUMNS = ["vehicle_type", "gearbox", "model", "fuel_type", "brand", "repaired"]
TARGET_COLUMN = "price"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed train/val dataset for car_price.")
    parser.add_argument(
        "--validated-csv",
        type=Path,
        default=Path("data/validated/autos_validated.csv"),
        help="Path to validated autos csv.",
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

    # Приводим ключевые числовые поля к числовому типу.
    for col in ["price", "registrationyear", "power", "kilometer", "registrationmonth"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Базовые фильтры по аномалиям.
    data = data[data["price"].between(100, 200000)]
    data = data[data["registrationyear"].between(1950, 2016)]
    data = data[data["power"].between(20, 1500)]
    data = data[data["kilometer"].between(0, 300000)]
    data = data[data["registrationmonth"].between(1, 12)]

    # Переименовываем в snake_case и добавляем age-признак.
    data = data.rename(
        columns={
            "registrationyear": "registration_year",
            "registrationmonth": "registration_month",
            "vehicletype": "vehicle_type",
            "fueltype": "fuel_type",
            "datecrawled": "date_crawled",
            "datecreated": "date_created",
            "numberofpictures": "number_of_pictures",
            "postalcode": "postal_code",
            "lastseen": "last_seen",
        }
    )
    data["car_age"] = 2016 - data["registration_year"]

    # Категориальные пропуски заполняем техническим значением.
    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna("unknown").astype(str)

    model_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    data = data[model_columns].dropna()
    return data


def main() -> None:
    # 1) Читаем validated слой и выполняем очистку/преобразование.
    args = parse_args()
    if not args.validated_csv.exists():
        raise FileNotFoundError(f"Validated csv not found: {args.validated_csv}")
    raw = pd.read_csv(args.validated_csv)
    dataset = clean_dataframe(raw)

    # 2) Формируем train/val выборки.
    train_df, val_df = train_test_split(dataset, test_size=args.test_size, random_state=args.random_state)

    # 3) Сохраняем датасеты и manifest признаков.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = write_table(train_df, args.output_dir / "train")
    val_path = write_table(val_df, args.output_dir / "val")

    feature_manifest = {
        "target": TARGET_COLUMN,
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved val: {val_path}")
    print(f"[build_dataset] Saved feature manifest: {manifest_path}")
    print(f"[build_dataset] Rows train={len(train_df)} val={len(val_df)}")


if __name__ == "__main__":
    main()
