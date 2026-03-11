from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analytical report for real_estate dataset.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed input directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Output directory for report.")
    return parser.parse_args()


def read_dataset(processed_dir: Path) -> pd.DataFrame:
    parquet_path = processed_dir / "dataset.parquet"
    csv_path = processed_dir / "dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Processed dataset not found (dataset.parquet/csv)")


def main() -> None:
    # 1) Загружаем очищенный датасет после build_dataset.
    args = parse_args()
    df = read_dataset(args.processed_dir)

    # 2) Считаем корреляции ключевых признаков с ценой.
    corr_cols = ["last_price", "total_area", "living_area", "kitchen_area", "rooms", "city_centers_nearest"]
    existing_corr_cols = [col for col in corr_cols if col in df.columns]
    corr_with_price = (
        df[existing_corr_cols]
        .corr(numeric_only=True)["last_price"]
        .drop(labels=["last_price"], errors="ignore")
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )

    # 3) Агрегируем среднюю цену по типу этажа.
    floor_type_price = (
        df.groupby("floor_type", dropna=False)["last_price"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    # 4) Формируем топ населенных пунктов по числу объявлений и цене за м2.
    locality_table = (
        df.groupby("locality_name", dropna=False)
        .agg(ads_count=("price_m", "count"), avg_price_m=("price_m", "mean"))
        .sort_values(by="ads_count", ascending=False)
        .head(10)
        .reset_index()
    )
    top_localities = locality_table.round({"avg_price_m": 2}).to_dict(orient="records")

    # 5) Считаем динамику цены по удаленности от центра для Санкт-Петербурга.
    spb = df[df["locality_name"] == "Санкт-Петербург"].copy()
    if not spb.empty and "city_centers_nearest_km" in spb.columns:
        spb_price_by_km_raw = (
            spb.groupby("city_centers_nearest_km", dropna=False)["last_price"]
            .mean()
            .sort_index()
            .round(2)
            .to_dict()
        )
        spb_price_by_km = {str(k): float(v) for k, v in spb_price_by_km_raw.items() if pd.notna(k)}
    else:
        spb_price_by_km = {}

    # 6) Собираем и сохраняем единый аналитический отчет.
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "price_summary": {
            "mean_last_price": float(df["last_price"].mean()),
            "median_last_price": float(df["last_price"].median()),
            "mean_days_exposition": float(df["days_exposition"].mean()),
            "median_days_exposition": float(df["days_exposition"].median()),
        },
        "corr_with_last_price": corr_with_price,
        "avg_price_by_floor_type": floor_type_price,
        "top_10_localities_by_ads": top_localities,
        "spb_avg_price_by_center_distance_km": spb_price_by_km,
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.artifacts_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[analyze] Saved report: {report_path}")
    print(f"[analyze] Rows: {len(df)}")
    print(f"[analyze] Mean price: {df['last_price'].mean():.2f}")


if __name__ == "__main__":
    main()
