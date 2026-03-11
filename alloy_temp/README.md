# Предсказание температуры сплава

## Используемый стек

![Python](https://img.shields.io/badge/-Python-blue)
![Pandas](https://img.shields.io/badge/-Pandas-blue)
![NumPy](https://img.shields.io/badge/-NumPy-yellow)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-orange)
![SQLite](https://img.shields.io/badge/-SQLite-003B57)
![PyArrow](https://img.shields.io/badge/-PyArrow-lightgrey)
![Joblib](https://img.shields.io/badge/-Joblib-teal)
![Requests](https://img.shields.io/badge/-Requests-green)
![PowerShell](https://img.shields.io/badge/-PowerShell-5391FE)

## Цель проекта

Построить воспроизводимый пайплайн для прогноза конечной температуры плавки:
от загрузки технологических данных из SQLite до обучения baseline-модели
и сохранения итоговых метрик.

## Данные

Источник данных: SQLite база `alloy_temp.db`.

Используемые таблицы:

- `data_arc` - параметры нагрева дугой (мощность и длительность этапов).
- `data_bulk` - объемы сыпучих материалов.
- `data_wire` - объемы проволочных материалов.
- `data_gas` - расход газа.
- `data_temp` - замеры температуры (время и значение).

## Этапы пайплайна

1. `ingest`  
   Скачивает `alloy_temp.db` в `data/raw` и создает `manifest.json`.
2. `validate`  
   Проверяет наличие обязательных таблиц и колонок, а также валидность температурных значений.
3. `build_dataset`  
   Строит агрегированные признаки по каждой плавке (`key`) и формирует train/val выборки.
4. `train`  
   Обучает `RandomForestRegressor`, считает `MAE` и `R2`, сохраняет модель и метрики.

## Структура проекта

```text
alloy_temp/
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
└── alloy_temp.ipynb
```

## Быстрый запуск

```powershell
pip install -r requirements.txt
./run_pipeline.ps1
```

Если нужен другой источник БД:

```powershell
./run_pipeline.ps1 -DataUrl "https://your-url/alloy_temp.db"
```

## Итоговые артефакты

- `data/raw/alloy_temp.db` - исходная база данных.
- `data/validated/quality_report.json` - отчет по проверкам качества.
- `data/processed/train.parquet` и `data/processed/val.parquet` - подготовленные выборки.
- `data/processed/feature_manifest.json` - список признаков и целевой переменной.
- `artifacts/model.joblib` - обученная baseline-модель.
- `artifacts/metrics.json` - метрики запуска (`MAE`, `R2`, важность признаков).
