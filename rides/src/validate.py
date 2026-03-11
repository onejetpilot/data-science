from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


REQUIRED_SCHEMAS = {
    "users_go.csv": {"user_id", "name", "age", "city", "subscription_type"},
    "rides_go.csv": {"user_id", "distance", "duration", "date"},
    "subscriptions_go.csv": {"subscription_type", "minute_price", "start_ride_price", "subscription_fee"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw rides files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory with raw CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Directory for validated files.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем, что все исходные файлы существуют и содержат обязательные поля.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"status": "ok", "files": {}}

    for file_name, required_columns in REQUIRED_SCHEMAS.items():
        input_path = args.raw_dir / file_name
        if not input_path.exists():
            raise FileNotFoundError(f"Required file not found: {input_path}")

        df = pd.read_csv(input_path)
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{file_name}: missing columns: {missing}")

        # 2) Копируем проверенные файлы в validated слой.
        output_path = args.output_dir / file_name
        shutil.copyfile(input_path, output_path)
        report["files"][file_name] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_columns": missing,
            "null_cells": int(df.isna().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum()),
        }
        print(f"[validate] OK: {file_name} ({len(df)} rows)")

    # 3) Сохраняем итоговый quality report.
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
