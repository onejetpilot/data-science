from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for rides.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"), help="Validated input directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Processed output directory.")
    return parser.parse_args()


def main() -> None:
    # 1) Загружаем валидированные таблицы пользователей, поездок и подписок.
    args = parse_args()
    users = pd.read_csv(args.validated_dir / "users_go.csv")
    rides = pd.read_csv(args.validated_dir / "rides_go.csv")
    subs = pd.read_csv(args.validated_dir / "subscriptions_go.csv")

    # 2) Подготавливаем поездки: дата в datetime и номер месяца.
    rides["date"] = pd.to_datetime(rides["date"], errors="coerce")
    rides = rides.dropna(subset=["date"])
    rides["month"] = rides["date"].dt.month

    # 3) Объединяем таблицы как в ноутбуке и округляем duration вверх.
    df = users.merge(subs, on="subscription_type", how="left")
    df = rides.merge(df, on="user_id", how="left")
    df["duration"] = np.ceil(pd.to_numeric(df["duration"], errors="coerce"))
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["duration", "distance", "subscription_type"])

    # 4) Строим помесячную агрегацию по пользователю для подсчета выручки.
    monthly = (
        df.groupby(
            ["month", "user_id", "minute_price", "start_ride_price", "subscription_fee", "subscription_type"],
            as_index=False,
        )
        .agg(distance=("distance", "sum"), duration=("duration", "sum"), rides=("user_id", "count"))
    )
    monthly["revenue"] = (
        monthly["start_ride_price"] * monthly["rides"]
        + monthly["subscription_fee"]
        + monthly["duration"] * monthly["minute_price"]
    )

    # 5) Сохраняем оба processed датасета и manifest.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = args.output_dir / "rides_enriched.parquet"
    monthly_path = args.output_dir / "monthly_user_metrics.parquet"
    try:
        df.to_parquet(enriched_path, index=False)
        monthly.to_parquet(monthly_path, index=False)
    except Exception:
        enriched_path = args.output_dir / "rides_enriched.csv"
        monthly_path = args.output_dir / "monthly_user_metrics.csv"
        df.to_csv(enriched_path, index=False)
        monthly.to_csv(monthly_path, index=False)

    manifest = {
        "enriched_dataset_path": str(enriched_path.resolve()),
        "monthly_dataset_path": str(monthly_path.resolve()),
        "rows_enriched": int(len(df)),
        "rows_monthly": int(len(monthly)),
        "target_metric": "revenue",
        "subscription_values": sorted(df["subscription_type"].dropna().astype(str).unique().tolist()),
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved enriched dataset: {enriched_path}")
    print(f"[build_dataset] Saved monthly dataset: {monthly_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Rows enriched: {len(df)}")
    print(f"[build_dataset] Rows monthly: {len(monthly)}")


if __name__ == "__main__":
    main()
