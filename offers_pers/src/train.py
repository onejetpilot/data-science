from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offers_pers model with notebook-like pipeline.")
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


def main() -> None:
    # 1) Загружаем данные и отделяем target.
    args = parse_args()
    df_2 = read_dataset(args.processed_dir)
    X = df_2.drop(columns=["покупательская_активность"]).set_index("id")
    y = pd.to_numeric(df_2["покупательская_активность"], errors="coerce").astype(int)

    # 2) Делим train/test как в ноутбуке.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3) Собираем preprocessor и финальный pipeline.
    ohe_columns = ["разрешить_сообщать", "популярная_категория"]
    ord_columns = ["тип_сервиса"]
    num_columns = X_train.select_dtypes(include="number").columns.tolist()

    ohe_pipe = Pipeline([("ohe", OneHotEncoder(drop="first", handle_unknown="error"))])
    ord_pipe = Pipeline([("ord", OrdinalEncoder(categories=[["стандарт", "премиум"]]))])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("num", StandardScaler())])

    data_preprocessor = ColumnTransformer(
        [
            ("ohe", ohe_pipe, ohe_columns),
            ("num", num_pipe, num_columns),
            ("ord", ord_pipe, ord_columns),
        ],
        remainder="passthrough",
    )

    pipe_final = Pipeline(
        [
            ("preprocessor", data_preprocessor),
            ("models", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    # 4) Диапазоны гиперпараметров как в ноутбуке.
    param_grid = [
        {
            "models": [KNeighborsClassifier()],
            "models__n_neighbors": range(1, 20),
            "models__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "preprocessor__num": [StandardScaler(), MinMaxScaler()],
        },
        {
            "models": [DecisionTreeClassifier(random_state=RANDOM_STATE)],
            "models__max_depth": range(2, 11),
            "models__min_samples_split": range(2, 10),
            "preprocessor__num": [StandardScaler(), MinMaxScaler(), "passthrough"],
        },
        {
            "models": [SVC(random_state=RANDOM_STATE, probability=True)],
            "models__kernel": ["linear", "rbf"],
            "models__C": range(1, 10, 1),
            "preprocessor__num": [StandardScaler(), MinMaxScaler(), "passthrough"],
        },
        {
            "models": [LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)],
            "models__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "preprocessor__num": [StandardScaler(), MinMaxScaler(), "passthrough"],
        },
    ]

    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        n_iter=15,
        scoring=["recall", "precision", "roc_auc"],
        refit="recall",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    randomized_search.fit(X_train, y_train)

    # 5) Считаем метрики на тесте и сохраняем артефакты.
    y_pred = randomized_search.predict(X_test)
    y_proba = randomized_search.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, zero_division=0)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    joblib.dump(randomized_search.best_estimator_, model_path)

    best_params_json = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v) for k, v in randomized_search.best_params_.items()}

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": str(type(randomized_search.best_estimator_.named_steps["models"]).__name__),
        "best_params": best_params_json,
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_roc_auc": float(roc_auc),
        "classification_report": report,
        "rows_total": int(len(df_2)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best estimator: {randomized_search.best_estimator_}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Recall: {recall:.4f}")
    print(f"[train] Precision: {precision:.4f}")
    print(f"[train] ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
