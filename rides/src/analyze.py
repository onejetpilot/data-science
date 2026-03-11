from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st


TARGET_DISTANCE_M = 3130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis report for rides project.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed input directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Output directory for analysis artifacts.")
    return parser.parse_args()


def read_table(processed_dir: Path, base_name: str) -> pd.DataFrame:
    parquet_path = processed_dir / f"{base_name}.parquet"
    csv_path = processed_dir / f"{base_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Processed file not found: {base_name}.parquet/csv")


def safe_float(value: float) -> float:
    return float(value) if pd.notna(value) else float("nan")


def main() -> None:
    # 1) Загружаем processed таблицы после build_dataset.
    args = parse_args()
    df = read_table(args.processed_dir, "rides_enriched")
    monthly = read_table(args.processed_dir, "monthly_user_metrics")

    # 2) Выделяем выборки по типу подписки.
    ultra = df[df["subscription_type"] == "ultra"]
    free = df[df["subscription_type"] == "free"]
    monthly_ultra = monthly[monthly["subscription_type"] == "ultra"]
    monthly_free = monthly[monthly["subscription_type"] == "free"]

    # 3) Считаем основные метрики продукта.
    city_freq = (
        df.groupby("city", dropna=False)["user_id"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )
    subscription_share = (df.groupby("subscription_type").size() / len(df)).to_dict()

    # 4) Проверяем гипотезу о длительности поездок (ultra > free).
    t_duration = st.ttest_ind(ultra["duration"], free["duration"], alternative="greater")

    # 5) Проверяем гипотезу о дистанции для ultra (mean > 3130 м).
    t_distance = st.ttest_1samp(ultra["distance"], TARGET_DISTANCE_M, alternative="greater")

    # 6) Проверяем гипотезу о выручке (ultra > free) по месячным метрикам.
    t_revenue = st.ttest_ind(monthly_ultra["revenue"], monthly_free["revenue"], alternative="greater")

    # 7) Собираем финальный JSON-отчет.
    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows_enriched": int(len(df)),
        "rows_monthly": int(len(monthly)),
        "summary": {
            "top_10_city_frequency": {str(k): int(v) for k, v in city_freq.items()},
            "subscription_share": {str(k): safe_float(v) for k, v in subscription_share.items()},
            "avg_duration_ultra": safe_float(ultra["duration"].mean()),
            "avg_duration_free": safe_float(free["duration"].mean()),
            "avg_distance_ultra": safe_float(ultra["distance"].mean()),
            "avg_revenue_ultra": safe_float(monthly_ultra["revenue"].mean()),
            "avg_revenue_free": safe_float(monthly_free["revenue"].mean()),
        },
        "hypotheses": {
            "duration_ultra_gt_free": {
                "statistic": safe_float(t_duration.statistic),
                "p_value": safe_float(t_duration.pvalue),
            },
            "distance_ultra_gt_3130m": {
                "threshold_m": TARGET_DISTANCE_M,
                "statistic": safe_float(t_distance.statistic),
                "p_value": safe_float(t_distance.pvalue),
            },
            "revenue_ultra_gt_free": {
                "statistic": safe_float(t_revenue.statistic),
                "p_value": safe_float(t_revenue.pvalue),
            },
        },
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.artifacts_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[analyze] Saved report: {report_path}")
    print(f"[analyze] Rows enriched: {len(df)}")
    print(f"[analyze] Avg revenue ultra: {monthly_ultra['revenue'].mean():.2f}")
    print(f"[analyze] Avg revenue free: {monthly_free['revenue'].mean():.2f}")


if __name__ == "__main__":
    main()
