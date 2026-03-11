from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "first_temp",
    "temp_measure_count",
    "temp_time_span_min",
    "arc_active_sum",
    "arc_reactive_sum",
    "arc_power_mean",
    "arc_duration_min_sum",
    "arc_heat_count",
    "bulk_total",
    "wire_total",
    "gas_1_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val dataset from alloy_temp sqlite.")
    parser.add_argument("--db-path", type=Path, default=Path("data/raw/alloy_temp.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _write_table(df: pd.DataFrame, base_path: Path) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def build_arc_features(conn: sqlite3.Connection) -> pd.DataFrame:
    arc = pd.read_sql_query(
        """
        SELECT
            key,
            `Начало нагрева дугой` AS arc_start,
            `Конец нагрева дугой` AS arc_end,
            `Активная мощность` AS active_power,
            `Реактивная мощность` AS reactive_power
        FROM data_arc
        """,
        conn,
    )
    arc["arc_start"] = pd.to_datetime(arc["arc_start"], errors="coerce")
    arc["arc_end"] = pd.to_datetime(arc["arc_end"], errors="coerce")
    arc["active_power"] = pd.to_numeric(arc["active_power"], errors="coerce")
    arc["reactive_power"] = pd.to_numeric(arc["reactive_power"], errors="coerce")

    arc["arc_duration_min"] = (arc["arc_end"] - arc["arc_start"]).dt.total_seconds() / 60.0
    arc["arc_duration_min"] = arc["arc_duration_min"].clip(lower=0).fillna(0)
    arc["arc_power"] = np.sqrt(arc["active_power"].fillna(0) ** 2 + arc["reactive_power"].fillna(0) ** 2)

    return (
        arc.groupby("key", as_index=False)
        .agg(
            arc_active_sum=("active_power", "sum"),
            arc_reactive_sum=("reactive_power", "sum"),
            arc_power_mean=("arc_power", "mean"),
            arc_duration_min_sum=("arc_duration_min", "sum"),
            arc_heat_count=("key", "count"),
        )
        .fillna(0)
    )


def build_sum_features(conn: sqlite3.Connection, table_name: str, prefix: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    value_cols = [c for c in df.columns if c != "key"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[f"{prefix}_total"] = df[value_cols].sum(axis=1)
    return df.groupby("key", as_index=False).agg(**{f"{prefix}_total": (f"{prefix}_total", "sum")})


def build_gas_features(conn: sqlite3.Connection) -> pd.DataFrame:
    gas = pd.read_sql_query("SELECT key, `Газ 1` AS gas_1 FROM data_gas", conn)
    gas["gas_1"] = pd.to_numeric(gas["gas_1"], errors="coerce").fillna(0)
    return gas.groupby("key", as_index=False).agg(gas_1_sum=("gas_1", "sum"))


def build_temp_features(conn: sqlite3.Connection) -> pd.DataFrame:
    temp = pd.read_sql_query("SELECT key, `Время замера` AS measure_time, `Температура` AS temperature FROM data_temp", conn)
    temp["measure_time"] = pd.to_datetime(temp["measure_time"], errors="coerce")
    temp["temperature"] = pd.to_numeric(temp["temperature"], errors="coerce")
    temp = temp.dropna(subset=["measure_time", "temperature"]).sort_values(["key", "measure_time"])

    grouped = temp.groupby("key", as_index=False).agg(
        first_time=("measure_time", "first"),
        last_time=("measure_time", "last"),
        first_temp=("temperature", "first"),
        target_temp=("temperature", "last"),
        temp_measure_count=("temperature", "count"),
    )
    grouped = grouped[grouped["temp_measure_count"] >= 2].copy()
    grouped["temp_time_span_min"] = (grouped["last_time"] - grouped["first_time"]).dt.total_seconds() / 60.0
    grouped["temp_time_span_min"] = grouped["temp_time_span_min"].fillna(0).clip(lower=0)
    return grouped[["key", "first_temp", "target_temp", "temp_measure_count", "temp_time_span_min"]]


def main() -> None:
    # 1) Открываем БД и строим блочные признаки из каждой таблицы.
    args = parse_args()
    if not args.db_path.exists():
        raise FileNotFoundError(f"DB not found: {args.db_path}")

    conn = sqlite3.connect(args.db_path)
    try:
        temp_features = build_temp_features(conn)
        arc_features = build_arc_features(conn)
        bulk_features = build_sum_features(conn, "data_bulk", "bulk")
        wire_features = build_sum_features(conn, "data_wire", "wire")
        gas_features = build_gas_features(conn)
    finally:
        conn.close()

    # 2) Объединяем блоки по key, заполняем пропуски нулями и готовим train/val.
    dataset = temp_features.merge(arc_features, on="key", how="left")
    dataset = dataset.merge(bulk_features, on="key", how="left")
    dataset = dataset.merge(wire_features, on="key", how="left")
    dataset = dataset.merge(gas_features, on="key", how="left")
    dataset = dataset.fillna(0)

    train_df, val_df = train_test_split(dataset, test_size=args.test_size, random_state=args.random_state)

    # 3) Сохраняем датасет и список признаков.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _write_table(train_df, args.output_dir / "train")
    val_path = _write_table(val_df, args.output_dir / "val")

    feature_manifest = {"target": "target_temp", "features": FEATURE_COLUMNS}
    feature_path = args.output_dir / "feature_manifest.json"
    feature_path.write_text(json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved val: {val_path}")
    print(f"[build_dataset] Saved feature manifest: {feature_path}")
    print(f"[build_dataset] Rows train={len(train_df)} val={len(val_df)}")


if __name__ == "__main__":
    main()
