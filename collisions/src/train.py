from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline for collisions.")
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
    # 1) Загружаем train/val таблицы и manifest признаков.
    args = parse_args()
    train_df = read_table(args.processed_dir / "train")
    val_df = read_table(args.processed_dir / "val")

    manifest_path = args.processed_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest["target"]
    feature_columns = manifest["numeric_features"] + manifest["binary_features"] + manifest["categorical_features"]
    cat_features = manifest["categorical_features"]

    # 2) Обучаем CatBoostClassifier.
    x_train = train_df[feature_columns]
    y_train = train_df[target]
    x_val = val_df[feature_columns]
    y_val = val_df[target]

    model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.random_state,
        verbose=False,
    )
    model.fit(x_train, y_train, cat_features=cat_features)

    # 3) Считаем метрики классификации.
    probs = model.predict_proba(x_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, probs)
    f1 = f1_score(y_val, preds, pos_label=1)
    precision = precision_score(y_val, preds, pos_label=1)
    recall = recall_score(y_val, preds, pos_label=1)

    # 4) Сохраняем модель и метрики.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.cbm"
    metrics_path = args.artifacts_dir / "metrics.json"
    model.save_model(str(model_path))

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "CatBoostClassifier",
        "target": target,
        "features": feature_columns,
        "categorical_features": cat_features,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Validation ROC-AUC: {roc_auc:.4f}")
    print(f"[train] Validation F1: {f1:.4f}")
    print(f"[train] Validation Precision: {precision:.4f}")
    print(f"[train] Validation Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
