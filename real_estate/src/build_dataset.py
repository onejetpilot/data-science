from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed dataset for real_estate.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"), help="Validated input directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory.")
    return parser.parse_args()


def to_floor_type(row: pd.Series) -> str:
    floor = row["floor"]
    floors_total = row["floors_total"]
    if floor == 1:
        return "первый"
    if floor == floors_total:
        return "последний"
    return "другой"


def fill_by_locality_median(df: pd.DataFrame, column: str) -> pd.Series:
    locality_median = df.groupby("locality_name")[column].transform("median")
    return df[column].fillna(locality_median).fillna(df[column].median())


def main() -> None:
    # 1) Загружаем валидированный датасет.
    args = parse_args()
    df = pd.read_csv(args.validated_dir / "real_estate_data.csv", sep="\t")

    # 2) Приводим названия колонок к виду из ноутбука.
    if "cityCenters_nearest" in df.columns:
        df = df.rename(columns={"cityCenters_nearest": "city_centers_nearest"})

    # 3) Заполняем пропуски по базовым правилам из ноутбука.
    df["floors_total"] = df["floors_total"].fillna(df["floor"] + 1)
    df["is_apartment"] = df["is_apartment"].fillna(False)
    df["locality_name"] = df["locality_name"].fillna("Другой")

    # 4) Заполняем living_area средним по числу комнат.
    living_mean_by_rooms = df.groupby("rooms")["living_area"].transform("mean")
    df["living_area"] = df["living_area"].fillna(living_mean_by_rooms)

    # 5) Заполняем расстояния по медиане населенного пункта.
    df["airports_nearest"] = fill_by_locality_median(df, "airports_nearest")
    df["city_centers_nearest"] = fill_by_locality_median(df, "city_centers_nearest")

    # 6) Заполняем прочие пропуски нулями или агрегатами.
    for col in ["parks_nearest", "parks_around3000", "ponds_around3000", "balcony", "ponds_nearest", "days_exposition"]:
        df[col] = df[col].fillna(0)
    df["kitchen_area"] = df["kitchen_area"].fillna(df["kitchen_area"].mean())
    df["ceiling_height"] = df["ceiling_height"].fillna(df["ceiling_height"].median())

    # 7) Приводим типы для ключевых колонок.
    df = df.astype(
        {
            "total_images": "int32",
            "rooms": "int32",
            "floors_total": "int32",
            "is_apartment": "bool",
            "balcony": "int32",
            "parks_around3000": "int32",
            "ponds_around3000": "int32",
            "days_exposition": "int32",
        }
    )
    df["first_day_exposition"] = pd.to_datetime(df["first_day_exposition"], errors="coerce")

    # 8) Добавляем инженерные признаки из ноутбука.
    df["price_m"] = (df["last_price"] / df["total_area"]).round(2)
    df["week_day"] = df["first_day_exposition"].dt.weekday
    df["month"] = df["first_day_exposition"].dt.month
    df["year"] = df["first_day_exposition"].dt.year
    df["city_centers_nearest_km"] = (df["city_centers_nearest"] / 1000).round().astype("Int64")
    df["floor_type"] = df.apply(to_floor_type, axis=1)

    # 9) Применяем фильтры выбросов как в анализе.
    df = df[df["total_area"] < 250]
    df = df[df["last_price"] < 25_000_000]
    df = df[df["city_centers_nearest"] < 60_000]
    df = df.drop_duplicates().reset_index(drop=True)

    # 10) Сохраняем processed датасет и manifest.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "dataset.parquet"
    try:
        df.to_parquet(data_path, index=False)
    except Exception:
        data_path = args.output_dir / "dataset.csv"
        df.to_csv(data_path, index=False)

    manifest = {
        "dataset_path": str(data_path.resolve()),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_like_column": "last_price",
        "key_features": [
            "total_area",
            "living_area",
            "kitchen_area",
            "rooms",
            "floor_type",
            "city_centers_nearest",
            "price_m",
        ],
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved dataset: {data_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Rows: {len(df)}")


if __name__ == "__main__":
    main()
