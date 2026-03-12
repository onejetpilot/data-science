from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DATA_FILE = "6_class_1.csv"
REQUIRED_COLUMNS = {
    "Temperature (K)",
    "Luminosity(L/Lo)",
    "Radius(R/Ro)",
    "Absolute magnitude(Mv)",
    "Star type",
    "Star color",
    "Spectral Class",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw star_temp dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory with raw CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Directory for validated files.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем наличие сырого файла.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / DATA_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # 2) Загружаем и валидируем обязательные колонки.
    df = pd.read_csv(raw_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 3) Проверяем базовое качество.
    null_cells = int(df.isna().sum().sum())
    duplicated_rows = int(df.duplicated().sum())
    if null_cells > 0:
        print(f"[validate] Warning: null cells found: {null_cells}")

    # 4) Копируем исходник в validated слой.
    validated_path = args.output_dir / DATA_FILE
    shutil.copyfile(raw_path, validated_path)

    # 5) Сохраняем quality report.
    report = {
        "status": "ok",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_columns": missing_columns,
        "null_cells": null_cells,
        "duplicated_rows": duplicated_rows,
        "temperature_min": float(pd.to_numeric(df["Temperature (K)"], errors="coerce").min()),
        "temperature_max": float(pd.to_numeric(df["Temperature (K)"], errors="coerce").max()),
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[validate] Rows: {len(df)}")
    print(f"[validate] Saved file: {validated_path}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
