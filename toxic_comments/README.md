# Классификация токсичных комментариев

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![LightGBM](https://img.shields.io/badge/-LightGBM-green)
![spaCy](https://img.shields.io/badge/-spaCy-lightblue)
![Requests](https://img.shields.io/badge/-Requests-green)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый mini-pipeline для классификации токсичных комментариев
с целевой метрикой `F1 >= 0.75`.

## Данные

Источник: `toxic_comments.csv` с колонками:

- `text` - текст комментария;
- `toxic` - целевой признак (0/1).

## Этапы пайплайна

1. `ingest`  
   Загружает исходный CSV в `data/raw` и сохраняет `manifest.json`.
2. `validate`  
   Проверяет обязательные колонки и корректность целевого признака.
3. `build_dataset`  
   Делает базовую нормализацию текста (lower + regex), формирует train/test split.
4. `train`  
   Сравнивает `LogisticRegression`, `LinearSVC`, `LightGBM` на TF-IDF (1-2 grams)
   по CV `F1`, выбирает лучшую и считает метрики на test.

## Структура проекта

```text
toxic_comments/
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
└── toxic_comments.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если файл уже есть локально:

```powershell
./run_pipeline.ps1 -LocalFile ".\toxic_comments.csv"
```

## Итоговые артефакты

- `data/validated/quality_report.json` - отчет по проверкам данных.
- `data/processed/train.parquet`, `data/processed/test.parquet` - train/test после подготовки.
- `data/processed/feature_manifest.json` - параметры split и признаки.
- `artifacts/model.joblib` - лучшая модель.
- `artifacts/metrics.json` - CV F1 и test F1 + проверка порога 0.75.
- `artifacts/test_predictions.csv` - прогнозы по тестовой выборке.
