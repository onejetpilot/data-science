from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from catboost import CatBoostRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline for car_price.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_table(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither parquet nor csv found for {base_path.name}")


def main() -> None:
    # 1) Загружаем train/val и manifest признаков.
    args = parse_args()
    train_df = read_table(args.processed_dir / "train")
    val_df = read_table(args.processed_dir / "val")

    manifest_path = args.processed_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest["target"]
    num_features = manifest["numeric_features"]
    cat_features = manifest["categorical_features"]
    feature_columns = num_features + cat_features

    # 2) Обучаем CatBoost с нативной обработкой категориальных признаков.
    x_train = train_df[feature_columns]
    y_train = train_df[target]
    x_val = val_df[feature_columns]
    y_val = val_df[target]

    model = CatBoostRegressor(
        iterations=600,
        depth=8,
        learning_rate=0.07,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.random_state,
        verbose=False,
    )
    model.fit(x_train, y_train, cat_features=cat_features)

    # 3) Считаем метрики на валидации.
    preds = model.predict(x_val)
    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)

    # 4) Сохраняем модель и метрики.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.cbm"
    metrics_path = args.artifacts_dir / "metrics.json"
    model.save_model(str(model_path))

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "CatBoostRegressor",
        "target": target,
        "features": feature_columns,
        "categorical_features": cat_features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "rmse": float(rmse),
        "mae": float(mae),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Validation RMSE: {rmse:.4f}")
    print(f"[train] Validation MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
