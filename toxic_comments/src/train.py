from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
MAX_TRAIN_ROWS = 15000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train toxic comments classifier with notebook-like setup.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def read_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    parquet_path = processed_dir / f"{split_name}.parquet"
    csv_path = processed_dir / f"{split_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Split not found: {split_name}.parquet/csv")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def main() -> None:
    args = parse_args()
    train_df = read_split(args.processed_dir, "train")
    test_df = read_split(args.processed_dir, "test")

    x_train_full = train_df["clean_text"].astype(str)
    y_train_full = train_df["toxic"].astype(int)
    x_test = test_df["clean_text"].astype(str)
    y_test = test_df["toxic"].astype(int)

    if len(train_df) > MAX_TRAIN_ROWS:
        sampled, _ = train_test_split(
            train_df[["clean_text", "toxic"]],
            train_size=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
            stratify=train_df["toxic"],
        )
        x_train = sampled["clean_text"].astype(str)
        y_train = sampled["toxic"].astype(int)
        sample_used = True
    else:
        x_train = x_train_full
        y_train = y_train_full
        sample_used = False

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    pipe_lr = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe_svm = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=20000,
                    C=0.25,
                    loss="squared_hinge",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv_scores = {
        "LogisticRegression": float(cross_val_score(pipe_lr, x_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()),
        "LinearSVC": float(cross_val_score(pipe_svm, x_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()),
    }
    best_name = max(cv_scores, key=cv_scores.get)
    best_pipeline = {
        "LogisticRegression": pipe_lr,
        "LinearSVC": pipe_svm,
    }[best_name]

    best_pipeline.fit(x_train_full, y_train_full)
    y_pred = best_pipeline.predict(x_test)
    test_f1 = float(f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    preds_path = args.artifacts_dir / "test_predictions.csv"

    joblib.dump(best_pipeline, model_path)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(preds_path, index=False)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "rows_train_used": int(len(x_train)),
        "sampled_training_used": bool(sample_used),
        "cv_f1_scores": cv_scores,
        "best_model_name": best_name,
        "test_f1": test_f1,
        "target_threshold_f1_0_75_met": bool(test_f1 >= 0.75),
        "classification_report": to_jsonable(report),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best model: {best_name}")
    print(f"[train] Test F1: {test_f1:.4f}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Saved predictions: {preds_path}")


if __name__ == "__main__":
    main()
