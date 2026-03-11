from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "weather_1",
    "road_surface",
    "lighting",
    "county_city_location",
    "county_location",
    "direction",
    "distance",
    "location_type",
    "road_condition_1",
    "at_fault",
    "insurance_premium",
    "cellphone_in_use",
    "vehicle_type",
    "vehicle_transmission",
    "vehicle_age",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate df_dtp.csv structure and target.")
    parser.add_argument("--raw-csv", type=Path, default=Path("data/raw/df_dtp.csv"), help="Path to raw df_dtp.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем входной файл и загружаем таблицу.
    args = parse_args()
    if not args.raw_csv.exists():
        raise FileNotFoundError(f"Raw csv not found: {args.raw_csv}")
    df = pd.read_csv(args.raw_csv)

    # 2) Проверяем схему и целевой признак.
    errors: list[str] = []
    warnings_list: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")

    if len(df) == 0:
        errors.append("dataset is empty")

    if "at_fault" in df.columns:
        y = pd.to_numeric(df["at_fault"], errors="coerce")
        invalid_target = int(y.isna().sum())
        unique_target = sorted(v for v in y.dropna().unique().tolist())
        if invalid_target > 0:
            errors.append(f"invalid at_fault values: {invalid_target}")
        if not set(unique_target).issubset({0, 1}):
            errors.append(f"at_fault must be binary 0/1, got: {unique_target}")
    else:
        invalid_target = None
        unique_target = []

    # 3) Сигнализируем о пропусках в признаках, но не падаем.
    feature_nulls = df.drop(columns=[c for c in ["at_fault"] if c in df.columns]).isna().sum().sum()
    if int(feature_nulls) > 0:
        warnings_list.append(f"missing feature values: {int(feature_nulls)}")

    # 4) Сохраняем quality report.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_csv": str(args.raw_csv.resolve()),
        "rows": int(len(df)),
        "required_columns": REQUIRED_COLUMNS,
        "invalid_target_values": invalid_target,
        "target_unique_values": unique_target,
        "warnings": warnings_list,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 5) При критических ошибках завершаем шаг исключением.
    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    # 6) Сохраняем валидированный CSV.
    validated_csv = args.output_dir / "df_dtp_validated.csv"
    df.to_csv(validated_csv, index=False)
    print(f"[validate] Saved validated csv: {validated_csv}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
