import os
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import ensure_spacy_model

app = FastAPI(title="Toxic Comment Classifier", version="0.1.0")

MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.joblib")
)


class CommentRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    toxic: bool
    score: Optional[float] = None


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена по пути {model_path}. Сначала запустите `python train.py`."
        )
    ensure_spacy_model()
    return joblib.load(model_path)


model = load_model()


@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CommentRequest):
    try:
        pred = model.predict([payload.text])[0]
        # LinearSVC не даёт вероятности; используем margin из decision_function
        margin = None
        if hasattr(model, "decision_function"):
            margin = float(model.decision_function([payload.text])[0])
        return PredictionResponse(toxic=bool(pred), score=margin)
    except Exception as exc:  # pragma: no cover - простая защита API от неожиданных ошибок
        raise HTTPException(status_code=500, detail=str(exc)) from exc

