# Персонализация предложений

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Joblib](https://img.shields.io/badge/-Joblib-teal)
![Requests](https://img.shields.io/badge/-Requests-green)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн персонализации предложений для интернет‑магазина:
на основе поведенческих и финансовых данных предсказать снижение
покупательской активности и выделить сегмент клиентов для адресных офферов.

## Данные

Источник данных:

- `market_file.csv`
- `market_money.csv`
- `market_time.csv`
- `money.csv`

Ключ объединения: `id`.

Целевая переменная:

- `покупательская_активность` (`0` = прежний уровень, `1` = снизилась).

## Этапы пайплайна

1. `ingest`  
   Скачивает исходные CSV в `data/raw`.
2. `validate`  
   Проверяет наличие файлов, схем и базовую целостность данных.
3. `build_dataset`  
   Повторяет предобработку ноутбука: исправления категорий, pivot периодов,
   фильтры по выручке, объединение таблиц и подготовка финального датасета.
4. `train`  
   Обучает модели через `RandomizedSearchCV` с `refit='recall'` и сохраняет лучшую.

## Структура проекта

```text
offers_pers/
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── build_dataset.py
│   └── train.py
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
├── artifacts/
├── run_pipeline.ps1
├── requirements.txt
└── offers_pers.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

## Итоговые артефакты

- `data/validated/quality_report.json` — отчет по проверкам данных.
- `data/processed/dataset.parquet` — обработанный датасет.
- `data/processed/feature_manifest.json` — описание target и признаков.
- `artifacts/model.joblib` — лучшая модель пайплайна.
- `artifacts/metrics.json` — тестовые метрики (`recall`, `precision`, `roc_auc`) и параметры лучшей модели.
