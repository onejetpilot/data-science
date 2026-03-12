from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate taxi forecasting models.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def read_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    parquet_path = processed_dir / f"{split_name}.parquet"
    csv_path = processed_dir / f"{split_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    raise FileNotFoundError(f"Split not found: {split_name}.parquet/csv")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    args = parse_args()
    train = read_split(args.processed_dir, "train")
    test = read_split(args.processed_dir, "test")

    x_train = train.drop(columns=["num_orders"])
    y_train = train["num_orders"]
    x_test = test.drop(columns=["num_orders"])
    y_test = test["num_orders"]

    pred_previous = y_test.shift(1)
    pred_previous.iloc[0] = y_train.iloc[-1]
    rmse_constant = rmse(y_test.to_numpy(), pred_previous.to_numpy())

    tscv = TimeSeriesSplit(n_splits=5)

    catboost_pipeline = Pipeline(
        [("model", CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, allow_writing_files=False))]
    )
    catboost_params = {
        "model__depth": [4, 6, 8],
        "model__iterations": [100, 300],
        "model__learning_rate": [0.03, 0.1],
    }
    catboost_search = GridSearchCV(
        catboost_pipeline,
        param_grid=catboost_params,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    catboost_search.fit(x_train, y_train)

    rf_pipeline = Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE))])
    rf_params = {"model__n_estimators": [50, 100], "model__max_depth": [5, 10]}
    rf_search = GridSearchCV(
        rf_pipeline,
        param_grid=rf_params,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    rf_search.fit(x_train, y_train)

    linreg_pipeline = Pipeline([("model", LinearRegression())])
    linreg_pipeline.fit(x_train, y_train)
    linreg_cv = -float(
        np.mean(
            cross_val_score(
                linreg_pipeline,
                x_train,
                y_train,
                scoring="neg_root_mean_squared_error",
                cv=tscv,
            )
        )
    )

    cv_results = {
        "CatBoost": -float(catboost_search.best_score_),
        "RandomForest": -float(rf_search.best_score_),
        "LinearRegression": linreg_cv,
    }
    best_model_name = min(cv_results, key=cv_results.get)

    final_model = {
        "CatBoost": catboost_search.best_estimator_,
        "RandomForest": rf_search.best_estimator_,
        "LinearRegression": linreg_pipeline,
    }[best_model_name]
    y_pred = final_model.predict(x_test)
    rmse_model = rmse(y_test.to_numpy(), y_pred)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    preds_path = args.artifacts_dir / "test_predictions.csv"

    joblib.dump(final_model, model_path)
    pd.DataFrame({"datetime": y_test.index, "y_true": y_test.values, "y_pred": y_pred}).to_csv(preds_path, index=False)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "cv_rmse": cv_results,
        "best_model_name": best_model_name,
        "best_model_params": to_jsonable(
            catboost_search.best_params_
            if best_model_name == "CatBoost"
            else rf_search.best_params_
            if best_model_name == "RandomForest"
            else {}
        ),
        "test_rmse_model": rmse_model,
        "test_rmse_constant_baseline": rmse_constant,
        "target_threshold_rmse_48_met": bool(rmse_model <= 48),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best model: {best_model_name}")
    print(f"[train] Test RMSE model: {rmse_model:.2f}")
    print(f"[train] Test RMSE baseline: {rmse_constant:.2f}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Saved predictions: {preds_path}")


if __name__ == "__main__":
    main()
