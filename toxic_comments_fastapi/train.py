"""
Скрипт обучения классификатора токсичных комментариев.
- Скачивает датасет при его отсутствии.
- Обучает конвейер TF-IDF + LinearSVC со spaCy-лемматизацией.
- Сохраняет обученную модель в artifacts/model.joblib.
"""

import os
import pathlib
from typing import Tuple

import joblib
import pandas as pd
import requests
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from pipeline import build_pipeline, ensure_spacy_model

BASE_URL = "https://huggingface.co/datasets/onejetpilot/toxic_comments/resolve/main/"
DATA_FILE = "toxic_comments.csv"
ARTIFACT_PATH = pathlib.Path(__file__).parent / "artifacts" / "model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def download_dataset(local_path: str = DATA_FILE) -> str:
    """Скачать датасет, если файла нет локально."""
    if os.path.exists(local_path):
        print(f"Dataset found locally at {local_path}")
        return local_path

    url = BASE_URL + DATA_FILE
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Dataset saved to {local_path}")
    return local_path


def load_data() -> Tuple[pd.Series, pd.Series]:
    path = download_dataset()
    df = pd.read_csv(path)
    return df["text"], df["toxic"]


def main():
    print("Ensuring spaCy model is available...")
    ensure_spacy_model()

    print("Loading data...")
    X, y = load_data()

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 on holdout: {f1:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    print(f"Saved trained pipeline to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

