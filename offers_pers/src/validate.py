from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "market_file.csv": {"id", "Покупательская активность", "Тип сервиса", "Разрешить сообщать"},
    "market_money.csv": {"id", "Период", "Выручка"},
    "market_time.csv": {"id", "Период", "минут"},
    "money.csv": {"id", "Прибыль"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw files for offers_pers.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Validated data directory.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем наличие файлов и обязательных колонок.
    args = parse_args()
    errors: list[str] = []
    warnings_list: list[str] = []
    stats: dict[str, int] = {}

    for filename, required_cols in REQUIRED.items():
        path = args.raw_dir / filename
        if not path.exists():
            errors.append(f"missing file: {filename}")
            continue

        df = pd.read_csv(path)
        stats[filename] = int(len(df))
        missing_cols = sorted(required_cols.difference(df.columns))
        if missing_cols:
            errors.append(f"{filename} missing columns: {missing_cols}")
        if len(df) == 0:
            errors.append(f"{filename} is empty")
        if df.duplicated().sum() > 0:
            warnings_list.append(f"{filename} contains duplicated rows")

    # 2) Сохраняем отчет валидации.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_dir": str(args.raw_dir.resolve()),
        "rows": stats,
        "warnings": warnings_list,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) Падаем только при критических ошибках.
    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    # 4) Копируем валидированные CSV в data/validated.
    for filename in REQUIRED:
        src = args.raw_dir / filename
        dst = args.output_dir / filename
        pd.read_csv(src).to_csv(dst, index=False)

    print(f"[validate] Saved report: {report_path}")
    print(f"[validate] Validation passed")


if __name__ == "__main__":
    main()
