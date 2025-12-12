# Toxic Comments FastAPI Service

Проект для обучения и запуска модели классификации токсичных комментариев через FastAPI и Docker.

## Структура
- `train.py` — скачивает датасет, обучает модель, сохраняет в `artifacts/model.joblib`.
- `pipeline.py` — лемматизация (spaCy pipe), сборка конвейера TF-IDF + LinearSVC.
- `app/main.py` — FastAPI приложение с эндпоинтами `/` и `/predict`.
- `requirements.txt` — зависимости.
- `Dockerfile` — сборка контейнера (ожидает готовый `artifacts/model.joblib`).

## Локальный запуск (без Docker)
```bash
cd toxic_comments_fastapi
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # если модель не скачана
python train.py  # скачает датасет, обучит и сохранит artifacts/model.joblib
uvicorn app.main:app --reload --port 8000
```
Проверка запроса:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "you are awesome"}'
```

## Сборка и запуск в Docker
1) Убедитесь, что файл `artifacts/model.joblib` уже создан командой `python train.py`.
2) Соберите образ и запустите:
```bash
cd toxic_comments_fastapi
docker build -t toxic-api .
docker run -p 8000:8000 toxic-api
```
API будет доступен на `http://localhost:8000`.

## Примечания
- Для обучения нужен интернет (скачать датасет и модель spaCy).
- Лемматизация работает пакетно через spaCy `pipe` для ускорения на всём датасете.
- LinearSVC использует `class_weight="balanced"` для учёта дисбаланса классов.
