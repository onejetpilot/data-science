from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train purchase_prediction models with notebook-like setup.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def read_dataset(processed_dir: Path) -> pd.DataFrame:
    parquet_path = processed_dir / "dataset.parquet"
    csv_path = processed_dir / "dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Processed dataset not found (dataset.parquet/csv)")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def main() -> None:
    # 1) Загружаем датасет и подготавливаем X/y.
    args = parse_args()
    df = read_dataset(args.processed_dir).copy()
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in processed dataset")
    if "client_id" in df.columns:
        df = df.drop(columns=["client_id"])

    y = pd.to_numeric(df["target"], errors="coerce")
    X = df.drop(columns=["target"]).copy()
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)

    # 2) Формируем списки признаков как в ноутбуке.
    bin_features = [col for col in X.columns if col.startswith("cat_")]
    cat_features = [col for col in ["event", "channel", "date_buy_dayofweek", "date_buy_month"] if col in X.columns]
    num_features = [col for col in ["price"] if col in X.columns]
    for col in cat_features:
        X[col] = X[col].astype(str).fillna("other")

    # 3) Делим на train/test со стратификацией.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 4) Собираем preprocessors и единый pipeline с переключаемым классификатором.
    num_encoder = Pipeline([("scaler", StandardScaler())])
    cat_encoder = Pipeline([("encoder", TargetEncoder())])
    lin_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_encoder, num_features),
            ("cat", cat_encoder, cat_features),
            ("bin", "passthrough", bin_features),
        ]
    )
    lgbm_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", cat_encoder, cat_features),
            ("bin", "passthrough", bin_features),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", lin_preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    # 5) Обучаем три GridSearchCV из ноутбука.
    param_grid_log = [
        {
            "preprocessor": [lin_preprocessor],
            "classifier": [LogisticRegression(class_weight="balanced", max_iter=1000)],
            "classifier__C": [0.01, 0.1, 1, 10],
        }
    ]
    param_grid_lgbm = [
        {
            "preprocessor": [lgbm_preprocessor],
            "classifier": [LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)],
            "classifier__n_estimators": [100, 300],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [5, 10],
        }
    ]
    # tuple здесь нужен для корректного clone() внутри GridSearchCV.
    catboost_cats = tuple(col for col in cat_features if col in X_train.columns)
    param_grid_cat = [
        {
            "preprocessor": ["passthrough"],
            "classifier": [
                CatBoostClassifier(
                    random_state=RANDOM_STATE,
                    cat_features=catboost_cats,
                    verbose=0,
                    auto_class_weights="Balanced",
                )
            ],
            "classifier__iterations": [100, 300],
        }
    ]

    grid_log = GridSearchCV(pipe, param_grid_log, scoring="roc_auc", cv=3, error_score="raise", n_jobs=-1)
    grid_lgbm = GridSearchCV(pipe, param_grid_lgbm, scoring="roc_auc", cv=3, error_score="raise", n_jobs=-1)
    grid_cat = GridSearchCV(pipe, param_grid_cat, scoring="roc_auc", cv=3, error_score="raise", n_jobs=-1)
    grid_log.fit(X_train, y_train)
    grid_lgbm.fit(X_train, y_train)
    grid_cat.fit(X_train, y_train)

    # 6) Считаем метрики на тесте и выбираем лучшую модель.
    grids = {"logreg": grid_log, "lgbm": grid_lgbm, "catboost": grid_cat}
    results: dict[str, dict[str, Any]] = {}
    for name, grid in grids.items():
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "roc_auc": float(roc_auc),
            "best_params": to_jsonable(grid.best_params_),
            "classification_report": to_jsonable(report),
        }
        print(f"[train] {name}: roc_auc={roc_auc:.4f}")

    best_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    best_estimator = grids[best_name].best_estimator_

    # 7) Сохраняем модель и сводные метрики.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    joblib.dump(best_estimator, model_path)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": best_name,
        "best_model_type": str(type(best_estimator.named_steps["classifier"]).__name__),
        "features": {
            "num_features": num_features,
            "cat_features": cat_features,
            "bin_features_count": len(bin_features),
        },
        "results": results,
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best model: {best_name}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
