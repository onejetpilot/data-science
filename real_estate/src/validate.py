from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "total_images",
    "last_price",
    "total_area",
    "first_day_exposition",
    "rooms",
    "ceiling_height",
    "floors_total",
    "living_area",
    "floor",
    "is_apartment",
    "studio",
    "open_plan",
    "kitchen_area",
    "balcony",
    "locality_name",
    "airports_nearest",
    "cityCenters_nearest",
    "parks_around3000",
    "parks_nearest",
    "ponds_around3000",
    "ponds_nearest",
    "days_exposition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw real_estate data file.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory with raw dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Directory for validated files.")
    return parser.parse_args()


def main() -> None:
    # 1) Загружаем сырой файл и проверяем его наличие.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / "real_estate_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # 2) Читаем данные и проверяем обязательные колонки.
    df = pd.read_csv(raw_path, sep="\t")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3) Копируем валидированный файл в слой validated.
    validated_path = args.output_dir / "real_estate_data.csv"
    shutil.copyfile(raw_path, validated_path)

    # 4) Сохраняем краткий quality report.
    report = {
        "status": "ok",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_columns": missing,
        "null_cells": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[validate] Rows: {len(df)}")
    print(f"[validate] Saved file: {validated_path}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
