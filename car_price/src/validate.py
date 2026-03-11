from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "DateCrawled",
    "Price",
    "VehicleType",
    "RegistrationYear",
    "Gearbox",
    "Power",
    "Model",
    "Kilometer",
    "RegistrationMonth",
    "FuelType",
    "Brand",
    "Repaired",
    "DateCreated",
    "NumberOfPictures",
    "PostalCode",
    "LastSeen",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate autos.csv structure and basic quality.")
    parser.add_argument("--raw-csv", type=Path, default=Path("data/raw/autos.csv"), help="Path to raw autos.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем наличие входного файла и читаем данные.
    args = parse_args()
    if not args.raw_csv.exists():
        raise FileNotFoundError(f"Raw csv not found: {args.raw_csv}")
    df = pd.read_csv(args.raw_csv)

    # 2) Проверяем обязательные колонки и базовую целостность.
    errors: list[str] = []
    warnings_list: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")

    row_count = int(len(df))
    if row_count == 0:
        errors.append("dataset is empty")

    price_num = pd.to_numeric(df.get("Price"), errors="coerce")
    invalid_price = int(price_num.isna().sum()) if "Price" in df.columns else None
    if invalid_price is not None and invalid_price > 0:
        errors.append(f"non-numeric Price values: {invalid_price}")

    if "RegistrationYear" in df.columns:
        reg_year_num = pd.to_numeric(df["RegistrationYear"], errors="coerce")
        invalid_reg_year = int(reg_year_num.isna().sum())
        if invalid_reg_year > 0:
            warnings_list.append(f"non-numeric RegistrationYear values: {invalid_reg_year}")
    else:
        invalid_reg_year = None

    # 3) Сохраняем quality report.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_csv": str(args.raw_csv.resolve()),
        "rows": row_count,
        "required_columns": REQUIRED_COLUMNS,
        "invalid_price_values": invalid_price,
        "invalid_registration_year_values": invalid_reg_year,
        "warnings": warnings_list,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 4) Прерываем пайплайн, если есть критические ошибки.
    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    # 5) Сохраняем валидированный CSV как есть (очистка будет на следующем шаге).
    validated_csv = args.output_dir / "autos_validated.csv"
    df.to_csv(validated_csv, index=False)
    print(f"[validate] Saved validated csv: {validated_csv}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
