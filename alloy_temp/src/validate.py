from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = {
    "data_arc": ["key", "Начало нагрева дугой", "Конец нагрева дугой", "Активная мощность", "Реактивная мощность"],
    "data_bulk": ["key"],
    "data_wire": ["key"],
    "data_gas": ["key", "Газ 1"],
    "data_temp": ["key", "Время замера", "Температура"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SQLite source for alloy_temp.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/raw/alloy_temp.db"),
        help="Path to raw sqlite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validated"),
        help="Directory for quality report.",
    )
    return parser.parse_args()


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    query = f"PRAGMA table_info('{table_name}')"
    return [row[1] for row in conn.execute(query).fetchall()]


def main() -> None:
    # 1) Проверяем наличие БД и открываем соединение.
    args = parse_args()
    if not args.db_path.exists():
        raise FileNotFoundError(f"DB not found: {args.db_path}")

    conn = sqlite3.connect(args.db_path)
    try:
        # 2) Проверяем обязательные таблицы и колонки.
        existing_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        errors: list[str] = []
        warnings_list: list[str] = []
        table_stats: dict[str, int] = {}

        for table_name, required_columns in REQUIRED_TABLES.items():
            if table_name not in existing_tables:
                errors.append(f"missing table: {table_name}")
                continue

            columns = table_columns(conn, table_name)
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                errors.append(f"table {table_name} missing columns: {missing_columns}")

            row_count = int(pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM '{table_name}'", conn)["cnt"].iloc[0])
            table_stats[table_name] = row_count
            if row_count == 0:
                errors.append(f"table {table_name} is empty")

        # 3) Дополнительно проверяем температуру: должна быть числовой после приведения.
        if "data_temp" in existing_tables:
            temp_df = pd.read_sql_query("SELECT key, `Температура` FROM data_temp", conn)
            temp_num = pd.to_numeric(temp_df["Температура"], errors="coerce")
            invalid_temp = int(temp_num.isna().sum())
            if invalid_temp > 0:
                warnings_list.append(f"data_temp invalid numeric values (will be dropped later): {invalid_temp}")
        else:
            invalid_temp = None

    finally:
        conn.close()

    # 4) Сохраняем quality report.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "db_path": str(args.db_path.resolve()),
        "required_tables": sorted(REQUIRED_TABLES.keys()),
        "table_rows": table_stats,
        "invalid_temperature_values": invalid_temp,
        "warnings": warnings_list,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Завершаем шаг ошибкой, если валидация не пройдена.
    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    print(f"[validate] Quality check passed: {report_path}")


if __name__ == "__main__":
    main()
