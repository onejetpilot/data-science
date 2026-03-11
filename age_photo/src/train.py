from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_COLUMNS = ["img_width", "img_height", "pixel_mean", "pixel_std"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline age model from tabular features.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with train and val datasets.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for model and metrics.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def read_table(base_path: Path) -> pd.DataFrame:
    # Читаем parquet, а если его нет, пробуем csv.
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither parquet nor csv found for {base_path.name}")


def main() -> None:
    # 1) Загружаем train/val таблицы из слоя processed.
    args = parse_args()
    train_df = read_table(args.processed_dir / "train")
    val_df = read_table(args.processed_dir / "val")

    # 2) Выделяем признаки и таргет.
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["real_age"]
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["real_age"]

    # 3) Обучаем baseline-регрессор и считаем MAE на валидации.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    mae = mean_absolute_error(y_val, preds)

    # 4) Сохраняем модель и метрики в artifacts.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    joblib.dump(model, model_path)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestRegressor",
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "mae": float(mae),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 5) Печатаем итог шага train.
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Validation MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
