from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
POINTS = 500
BEST_POINTS = 200
BUDGET = 10_000_000_000
UNIT_PROFIT = 450_000
BOOTSTRAP_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze well locations and estimate region profit/risk.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def read_region(processed_dir: Path, idx: int) -> pd.DataFrame:
    parquet_path = processed_dir / f"region_{idx}.parquet"
    csv_path = processed_dir / f"region_{idx}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Processed region file not found for index={idx}")


def fit_region(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    x = df[["f0", "f1", "f2"]]
    y = df["product"]
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.75, random_state=RANDOM_STATE)
    model = LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    results = pd.DataFrame({"product": y_val.values, "product_pred": preds})
    rmse = float(root_mean_squared_error(y_val, preds))
    return results, float(results["product_pred"].mean()), float(results["product"].mean()), rmse


def profit_calc(df: pd.DataFrame, col_pred: str = "product_pred", col_real: str = "product") -> tuple[float, float]:
    selected = df.sort_values(by=col_pred, ascending=False).head(BEST_POINTS)
    product_sum = float(selected[col_real].sum())
    profit = product_sum * UNIT_PROFIT - BUDGET
    return product_sum, float(profit)


def bootstrap_profit(df: pd.DataFrame, rng: np.random.Generator, n_samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, tuple[float, float], float]:
    profits = []
    n = len(df)
    for _ in range(n_samples):
        idx = rng.choice(n, size=POINTS, replace=True)
        sample = df.iloc[idx]
        _, profit = profit_calc(sample)
        profits.append(profit)
    arr = np.array(profits, dtype=float)
    mean_profit = float(arr.mean())
    lower, upper = np.percentile(arr, [2.5, 97.5])
    loss_risk = float((arr < 0).mean())
    return mean_profit, (float(lower), float(upper)), loss_risk


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(RANDOM_STATE)

    required_volume = BUDGET / (UNIT_PROFIT * BEST_POINTS)
    report: dict[str, object] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "constants": {
            "points": POINTS,
            "best_points": BEST_POINTS,
            "budget": BUDGET,
            "unit_profit": UNIT_PROFIT,
            "required_volume_per_well": required_volume,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "regions": {},
    }

    region_profits: dict[str, float] = {}
    for idx in [0, 1, 2]:
        df = read_region(args.processed_dir, idx)
        results, pred_mean, true_mean, rmse = fit_region(df)
        product_sum, profit = profit_calc(results)
        mean_profit, conf_int, risk = bootstrap_profit(results, rng=rng)

        region_key = f"region_{idx}"
        report["regions"][region_key] = {
            "rows": int(len(df)),
            "pred_mean": pred_mean,
            "true_mean": true_mean,
            "rmse": rmse,
            "top_200_product_sum": product_sum,
            "profit_top_200": profit,
            "bootstrap_mean_profit": mean_profit,
            "bootstrap_confidence_interval_95": [conf_int[0], conf_int[1]],
            "loss_risk": risk,
            "risk_below_2_5_percent": bool(risk < 0.025),
        }
        region_profits[region_key] = mean_profit

    # Выбираем регион с максимальной средней прибылью при ограничении риска.
    candidate_regions = {
        region: stats
        for region, stats in report["regions"].items()
        if bool(stats["risk_below_2_5_percent"])
    }
    if candidate_regions:
        recommended = max(candidate_regions, key=lambda name: float(candidate_regions[name]["bootstrap_mean_profit"]))
    else:
        recommended = max(region_profits, key=region_profits.get)
    report["recommended_region"] = recommended

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.artifacts_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[analyze] Recommended region: {recommended}")
    print(f"[analyze] Saved report: {report_path}")


if __name__ == "__main__":
    main()
