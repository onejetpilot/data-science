from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train startup success model with notebook-like setup.")
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
    raise FileNotFoundError(f"Processed split not found: {split_name}.parquet/csv")


def to_dataframe(x: Any) -> pd.DataFrame:
    return pd.DataFrame(x)


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
    train_df = read_split(args.processed_dir, "train").copy()
    test_df = read_split(args.processed_dir, "test").copy()
    if "status" not in train_df.columns:
        raise ValueError("Column 'status' not found in processed train split")

    # 1) Подготовка признаков и целевой переменной.
    numeric_features = train_df.select_dtypes(include=["number"]).columns.difference(["status"]).tolist()
    categorical_features = train_df.select_dtypes(include=["object"]).columns.difference(["status", "name"]).tolist()
    train_df[categorical_features] = train_df[categorical_features].astype("category")
    test_df[categorical_features] = test_df[categorical_features].astype("category")

    le = LabelEncoder()
    y = le.fit_transform(train_df["status"])
    x = train_df.drop(columns=["status", "name"], errors="ignore")
    x_test = test_df.drop(columns=["name"], errors="ignore")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2) Препроцессинг + pipeline, как в ноутбуке.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
            ("encoder", CatBoostEncoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("to_df", FunctionTransformer(to_dataframe, feature_names_out="one-to-one")),
            ("classifier", CatBoostClassifier(random_state=RANDOM_STATE, auto_class_weights="Balanced", silent=True)),
        ]
    )

    # 3) Подбор гиперпараметров (умеренный n_iter для воспроизводимого runtime).
    param_grid = [
        {
            "classifier": [
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    objective="binary",
                    verbose=-1,
                )
            ],
            "classifier__n_estimators": [100, 300, 500],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__num_leaves": [31, 63, 127],
            "classifier__max_depth": [5, 10, -1],
            "classifier__min_child_samples": [10, 20, 50],
        },
        {
            "classifier": [CatBoostClassifier(random_state=RANDOM_STATE, auto_class_weights="Balanced", silent=True)],
            "classifier__iterations": [50, 100, 200],
            "classifier__depth": [8, 10, 12],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__l2_leaf_reg": [1, 3, 5],
        },
    ]
    f1_scorer = make_scorer(f1_score, pos_label=0)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=24,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    random_search.fit(x_train, y_train)

    # 4) Метрики на валидации и дообучение на полном train.
    y_val_pred = random_search.predict(x_val)
    report_dict = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    best_pipeline = random_search.best_estimator_
    best_pipeline.fit(x, y)

    # 5) Предсказания для test и сохранение артефактов.
    y_test_pred = best_pipeline.predict(x_test)
    submit = test_df[["name"]].copy() if "name" in test_df.columns else pd.DataFrame({"name": range(len(test_df))})
    submit["status"] = le.inverse_transform(y_test_pred.astype(int))

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    submit_path = args.artifacts_dir / "submit_predictions.csv"

    joblib.dump(best_pipeline, model_path)
    submit.to_csv(submit_path, index=False)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows_train_total": int(len(x)),
        "rows_val": int(len(x_val)),
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "best_model_type": str(type(best_pipeline.named_steps["classifier"]).__name__),
        "best_params": to_jsonable(random_search.best_params_),
        "cv_best_score_f1_closed": float(random_search.best_score_),
        "validation_report": to_jsonable(report_dict),
        "submission_file": str(submit_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best model: {metrics['best_model_type']}")
    print(f"[train] CV f1(closed): {metrics['cv_best_score_f1_closed']:.4f}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Saved submission: {submit_path}")


if __name__ == "__main__":
    main()
