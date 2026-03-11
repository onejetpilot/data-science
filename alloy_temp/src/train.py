from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model for alloy_temp.")
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
    # 1) Загружаем train/val таблицы и feature manifest.
    args = parse_args()
    train_df = read_table(args.processed_dir / "train")
    val_df = read_table(args.processed_dir / "val")

    feature_path = args.processed_dir / "feature_manifest.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {feature_path}")
    feature_manifest = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_columns = feature_manifest["features"]
    target_col = feature_manifest["target"]

    # 2) Готовим матрицы признаков и обучаем baseline-модель.
    x_train = train_df[feature_columns]
    y_train = train_df[target_col]
    x_val = val_df[feature_columns]
    y_val = val_df[target_col]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    # 3) Считаем метрики на валидации.
    preds = model.predict(x_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # 4) Сохраняем модель, метрики и важности признаков.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    joblib.dump(model, model_path)

    feature_importance = dict(sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True))
    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestRegressor",
        "target": target_col,
        "features": feature_columns,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "mae": float(mae),
        "r2": float(r2),
        "feature_importance": feature_importance,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Validation MAE: {mae:.4f}")
    print(f"[train] Validation R2: {r2:.4f}")


if __name__ == "__main__":
    main()
